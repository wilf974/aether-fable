"""SeasonalMultiAgentFoodGrid — env V3 avec saisons, météo, food regen modulé.

Étend MultiAgentFoodGrid avec :
    - SeasonalConfig : période saisonnière + 4 facteurs lambda + range température
    - SeasonClock : phase ∈ [0, 1), saison courante (Spring/Summer/Autumn/Winter)
    - WeatherField : température 2D (gradient nord-sud + modulation saisonnière)
    - Food regen modulé par saison (I11) — printemps abondant, hiver rare
    - Metabolism modifié dans cellules froides (zones de danger passif)

H3 : un agent récurrent (LSTM) devrait apprendre des stratégies saisonnières
meilleures qu'un agent feedforward — testable en V3.5 avec DRQN MW_IA.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from aetherlife.guardrails.invariants import (
    cache_after_deposit,
    cache_after_withdrawal,
    child_birth_tick,
    child_generation,
    clamp_pos,
    clamp_temp,
    energy_after_build,
    energy_after_withdrawal,
    energy_no_food,
    energy_with_food,
    is_terminated,
    rest_energy_gain,
    season_phase,
    seasonal_lambda,
    step_reward,
)
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig, NestRecord
from aetherlife.world.food_grid import Action, _DELTAS
from aetherlife.world.multi_agent_grid import _AgentState
from aetherlife.world.biomes import (
    BiomeConfig, biome_params_for, generate_biome_map,
)
from aetherlife.world.competition import (
    CompetitionConfig, crowd_metabolism_factor,
)
from aetherlife.world.planting import PlantingConfig, PlantRecord
from aetherlife.world.reproduction import LineageEdge, ReproductionConfig
from aetherlife.world.traits import AgentTraits, TraitDistribution, TraitsConfig
from aetherlife.world.cooperative import CooperativeConfig, GatherSpot
from aetherlife.world.cooperative_metrics import (
    CooperativeMetricsConfig,
    CooperativeMetricsTracker,
)
from aetherlife.world.vocabulary import VocabularyConfig


class Season(IntEnum):
    SPRING = 0
    SUMMER = 1
    AUTUMN = 2
    WINTER = 3


@dataclass(frozen=True)
class SeasonalConfig:
    """Configuration saisonnière V3."""

    # Saisons
    season_period: int = 200            # 50 ticks par saison
    spring_lambda_factor: float = 1.5   # food regen abondante au printemps
    summer_lambda_factor: float = 1.0   # nominale
    autumn_lambda_factor: float = 1.2   # fruits secondaires
    winter_lambda_factor: float = 0.3   # rare en hiver

    # Météo
    temp_min: float = -10.0
    temp_max: float = 30.0
    temp_gradient_amplitude: float = 10.0  # spread nord-sud
    seasonal_amplitude: float = 15.0       # spread saisonnier (summer chaud / winter froid)

    # Métabolisme modifié si froid
    cold_threshold: float = 5.0            # cellules < 5°C
    cold_metabolism_factor: float = 1.5    # +50 % metabolism dans le froid

    def __post_init__(self) -> None:
        if self.season_period <= 0:
            raise ValueError(f"season_period doit être > 0 (got {self.season_period})")
        if self.temp_min >= self.temp_max:
            raise ValueError(
                f"temp_min ({self.temp_min}) doit être < temp_max ({self.temp_max})"
            )
        for f in (
            self.spring_lambda_factor, self.summer_lambda_factor,
            self.autumn_lambda_factor, self.winter_lambda_factor,
        ):
            if f < 0:
                raise ValueError(f"lambda_factor doit être ≥ 0 (got {f})")
        if self.cold_metabolism_factor < 1.0:
            raise ValueError(
                f"cold_metabolism_factor doit être >= 1 (got {self.cold_metabolism_factor})"
            )


@dataclass(frozen=True)
class SeasonalMultiAgentConfig:
    """Config complète V3 = MA + saisons."""

    rows: int = 32
    cols: int = 32
    n_agents: int = 16
    max_energy: float = 100.0
    start_energy: float = 50.0
    metabolism: float = 1.0
    food_value: float = 20.0
    death_penalty: float = 50.0
    initial_food_density: float = 0.05
    food_respawn_lambda: float = 1.0
    max_steps: int = 1000
    seasonal: SeasonalConfig = SeasonalConfig()
    reproduction: ReproductionConfig = ReproductionConfig()
    build: BuildConfig = BuildConfig()
    cache: CacheConfig = CacheConfig()
    planting: PlantingConfig = PlantingConfig()
    traits: TraitsConfig = TraitsConfig()  # V7 — héritage de biais comportementaux
    biomes: BiomeConfig = BiomeConfig()    # V8-B1.5 — niches écologiques
    competition: CompetitionConfig = CompetitionConfig()  # V8-B1.5 — densité locale
    vocabulary: VocabularyConfig = VocabularyConfig()  # V8-B2.0 — langage émergent
    cooperative: CooperativeConfig = CooperativeConfig()  # V8-C3 — coop

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows et cols doivent être > 0")
        if self.n_agents <= 0:
            raise ValueError(f"n_agents doit être > 0")
        if self.n_agents > self.rows * self.cols:
            raise ValueError(f"n_agents > rows*cols")
        if self.max_energy <= 0:
            raise ValueError("max_energy doit être > 0")
        if not (0 < self.start_energy <= self.max_energy):
            raise ValueError("start_energy doit être dans (0, max_energy]")
        if self.metabolism <= 0 or self.food_value <= 0:
            raise ValueError("metabolism et food_value doivent être > 0")
        if self.death_penalty < 0:
            raise ValueError("death_penalty doit être >= 0")
        if not (0.0 <= self.initial_food_density <= 1.0):
            raise ValueError("initial_food_density doit être dans [0, 1]")
        if self.food_respawn_lambda < 0:
            raise ValueError("food_respawn_lambda doit être >= 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps doit être > 0")

    @property
    def obs_dim(self) -> int:
        """obs = self + others + food + my_energy + season_phase + local_temp_normalized."""
        return 3 * self.rows * self.cols + 3


def current_season(phase: float) -> Season:
    """Phase ∈ [0, 1) → saison enum (4 quarts)."""
    if phase < 0.25:
        return Season.SPRING
    elif phase < 0.5:
        return Season.SUMMER
    elif phase < 0.75:
        return Season.AUTUMN
    else:
        return Season.WINTER


def get_seasonal_factor(season: Season, cfg: SeasonalConfig) -> float:
    """Retourne le facteur lambda pour la saison."""
    return {
        Season.SPRING: cfg.spring_lambda_factor,
        Season.SUMMER: cfg.summer_lambda_factor,
        Season.AUTUMN: cfg.autumn_lambda_factor,
        Season.WINTER: cfg.winter_lambda_factor,
    }[season]


def build_temperature_field(
    rows: int,
    cols: int,
    phase: float,
    cfg: SeasonalConfig,
) -> np.ndarray:
    """Construit le champ de température 2D pour le tick courant.

    - Gradient nord-sud (rangée 0 = froide, rangée H-1 = chaude).
    - Modulation saisonnière : summer +amplitude, winter -amplitude (sinusoïde).
    - Clampage final par I10.
    """
    base = (cfg.temp_min + cfg.temp_max) / 2.0  # ~10°C centre
    # Gradient nord-sud (+/- amplitude/2 du centre)
    row_factors = (
        np.linspace(-1.0, 1.0, rows) * (cfg.temp_gradient_amplitude / 2.0)
    )
    # Modulation saisonnière (cycle complet sur 1 phase, max summer phase=0.375)
    seasonal_offset = cfg.seasonal_amplitude * np.sin(
        2 * np.pi * (phase - 0.125)
    )
    temp_per_row = base + row_factors + seasonal_offset
    field = np.tile(temp_per_row[:, None], (1, cols))
    return np.vectorize(lambda t: clamp_temp(t, cfg.temp_min, cfg.temp_max))(field)


class SeasonalMultiAgentFoodGrid:
    """Env V3 saisonnier multi-agent — étend MultiAgentFoodGrid."""

    def __init__(self, cfg: SeasonalMultiAgentConfig | None = None) -> None:
        self.cfg = cfg or SeasonalMultiAgentConfig()
        self._food_mask: np.ndarray = np.zeros(
            (self.cfg.rows, self.cfg.cols), dtype=bool
        )
        self._temp_field: np.ndarray = np.zeros(
            (self.cfg.rows, self.cfg.cols), dtype=np.float32
        )
        # V8-B1.5 — biome map (init plain partout; généré via Voronoi à reset
        # si biomes.enabled)
        self._biome_map: np.ndarray = np.zeros(
            (self.cfg.rows, self.cfg.cols), dtype=np.int8
        )
        # V8-C3 — gather_spots (coopération)
        self._gather_spots: dict[tuple[int, int], GatherSpot] = {}
        self._gather_successes_total: int = 0
        self._gather_failures_total: int = 0
        # V8-C3 — observer télémétrie (clustering, delays, token entropy, chains)
        self._coop_metrics: CooperativeMetricsTracker = (
            CooperativeMetricsTracker(CooperativeMetricsConfig())
        )
        self._agents: list[_AgentState] = []
        self._step_count: int = 0
        self._env_rng: np.random.Generator = np.random.default_rng()
        self._spawn_rng: np.random.Generator = np.random.default_rng()
        self._placement_rng: np.random.Generator = np.random.default_rng()
        # V4 lineage tracking
        self._lineage: list[LineageEdge] = []
        self._next_agent_id: int = 0
        self._births_last_step: list[int] = []
        # V5 construction
        self._nests: dict[int, NestRecord] = {}
        self._builds_last_step: list[NestRecord] = []
        # V5 metrics
        self._nest_visits_total: int = 0
        self._rest_energy_gained_total: float = 0.0
        # V5.2 family metric
        self._family_nest_visits_total: int = 0
        # V5.3 cache state
        self._nest_food_stock: dict[int, float] = {}
        self._cache_deposits_total: int = 0
        self._cache_withdrawals_total: int = 0
        self._cache_energy_deposited_total: float = 0.0
        self._cache_energy_withdrawn_total: float = 0.0
        # V6 — plantation
        self._plants: dict[tuple[int, int], PlantRecord] = {}
        self._plants_planted_total: int = 0
        self._plants_matured_total: int = 0
        self._last_plant_tick: dict[int, int] = {}

    @property
    def n_actions(self) -> int:
        return len(Action)

    @property
    def n_states(self) -> int:
        return self.cfg.obs_dim

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def n_alive(self) -> int:
        return sum(1 for a in self._agents if a.alive)

    @property
    def n_dead(self) -> int:
        return sum(1 for a in self._agents if not a.alive)

    @property
    def food_count(self) -> int:
        return int(self._food_mask.sum())

    @property
    def food_mask(self) -> np.ndarray:
        return self._food_mask.copy()

    @property
    def biome_map(self) -> np.ndarray:
        """V8-B1.5 — 2D int8, biome_id par tile (0..3)."""
        return self._biome_map.copy()

    @property
    def gather_spots(self) -> dict[tuple[int, int], GatherSpot]:
        """V8-C3 — Dict (pos → GatherSpot) des spots actifs."""
        return dict(self._gather_spots)

    @property
    def gather_successes_total(self) -> int:
        return self._gather_successes_total

    @property
    def gather_failures_total(self) -> int:
        return self._gather_failures_total

    @property
    def coop_metrics(self) -> CooperativeMetricsTracker:
        """V8-C3 — Accès en lecture aux métriques coop pour reporting."""
        return self._coop_metrics

    @property
    def temperature_field(self) -> np.ndarray:
        return self._temp_field.copy()

    @property
    def phase(self) -> float:
        return season_phase(self._step_count, self.cfg.seasonal.season_period)

    @property
    def season(self) -> Season:
        return current_season(self.phase)

    @property
    def alive_agent_ids(self) -> list[int]:
        return [a.agent_id for a in self._agents if a.alive]

    @property
    def lineage(self) -> list[LineageEdge]:
        return list(self._lineage)

    @property
    def births_last_step(self) -> list[int]:
        return list(self._births_last_step)

    @property
    def n_births_total(self) -> int:
        return len(self._lineage)

    # V5 construction state
    @property
    def nests(self) -> dict[int, NestRecord]:
        return dict(self._nests)

    @property
    def n_nests(self) -> int:
        return len(self._nests)

    @property
    def builds_last_step(self) -> list[NestRecord]:
        return list(self._builds_last_step)

    @property
    def nest_positions(self) -> set[tuple[int, int]]:
        return {n.pos for n in self._nests.values()}

    @property
    def nest_visits_total(self) -> int:
        """Nombre cumulé de ticks où un agent est sur son propre nid."""
        return self._nest_visits_total

    @property
    def rest_energy_gained_total(self) -> float:
        """Somme cumulée de l'énergie regagnée via rest_bonus."""
        return self._rest_energy_gained_total

    @property
    def family_nest_visits_total(self) -> int:
        return self._family_nest_visits_total

    # V5.3 cache state
    @property
    def nest_food_stock(self) -> dict[int, float]:
        return dict(self._nest_food_stock)

    @property
    def total_cached_food(self) -> float:
        return sum(self._nest_food_stock.values())

    @property
    def cache_deposits_total(self) -> int:
        return self._cache_deposits_total

    @property
    def cache_withdrawals_total(self) -> int:
        return self._cache_withdrawals_total

    @property
    def cache_energy_deposited_total(self) -> float:
        return self._cache_energy_deposited_total

    @property
    def cache_energy_withdrawn_total(self) -> float:
        return self._cache_energy_withdrawn_total

    # V6 — plantation
    @property
    def plants(self) -> dict[tuple[int, int], PlantRecord]:
        return dict(self._plants)

    @property
    def n_plants(self) -> int:
        return len(self._plants)

    @property
    def plants_planted_total(self) -> int:
        return self._plants_planted_total

    @property
    def plants_matured_total(self) -> int:
        return self._plants_matured_total

    # V7 — traits
    @property
    def trait_distribution(self) -> TraitDistribution:
        """Snapshot moyenne/écart-type des traits sur la population vivante."""
        if not self.cfg.traits.enabled:
            return TraitDistribution()
        alive_traits = [
            a.traits for a in self._agents
            if a.alive and a.traits is not None
        ]
        return TraitDistribution.from_traits(alive_traits)

    def agent_traits(self, agent_id: int) -> AgentTraits | None:
        """Retourne les traits d'un agent, ou None si pas de traits."""
        try:
            return self.agent_state(agent_id).traits
        except KeyError:
            return None

    def can_plant_at(self, agent: _AgentState) -> bool:
        """V6 — vérifie si l'agent peut planter à sa position actuelle.

        V6.1 — requiert aussi `seeds >= seeds_required`.
        """
        pcfg = self.cfg.planting
        if not pcfg.enabled or not agent.alive:
            return False
        if agent.energy < pcfg.energy_threshold:
            return False
        if agent.seeds < pcfg.seeds_required:
            return False
        last = self._last_plant_tick.get(agent.agent_id, -10**9)
        if self._step_count - last < pcfg.cooldown_ticks:
            return False
        pos = agent.pos
        if self._food_mask[pos[0], pos[1]]:
            return False
        if pos in self._plants:
            return False
        if pos in self.nest_positions:
            return False
        return True

    def root_ancestor_of(self, agent_id: int) -> int:
        return self.agent_state(agent_id).root_ancestor_id

    def lineage_ids(self, root_id: int) -> list[int]:
        return [a.agent_id for a in self._agents if a.root_ancestor_id == root_id]

    def has_living_descendant(self, root_id: int) -> bool:
        return any(
            a.alive and a.root_ancestor_id == root_id for a in self._agents
        )

    def _accessible_nest_owner(self, agent: _AgentState) -> int | None:
        """V5.3 — owner_id du nid accessible sur la cellule de l'agent."""
        own_nest = self._nests.get(agent.agent_id)
        if own_nest is not None and own_nest.pos == agent.pos:
            return agent.agent_id
        if not self.cfg.build.family_inheritance:
            return None
        for nest in self._nests.values():
            if nest.pos != agent.pos:
                continue
            try:
                owner_root = self.agent_state(nest.owner_id).root_ancestor_id
            except KeyError:
                owner_root = nest.owner_id
            if owner_root == agent.root_ancestor_id:
                return nest.owner_id
        return None

    def agent_state(self, agent_id: int) -> _AgentState:
        for a in self._agents:
            if a.agent_id == agent_id:
                return a
        raise KeyError(f"agent_id={agent_id} not found")

    def _refresh_temperature(self) -> None:
        self._temp_field = build_temperature_field(
            self.cfg.rows, self.cfg.cols, self.phase, self.cfg.seasonal
        ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
        if seed is not None:
            self._env_rng = np.random.default_rng(seed)
            self._spawn_rng = np.random.default_rng(seed + 1)
            self._placement_rng = np.random.default_rng(seed + 2)
        self._step_count = 0
        self._food_mask = np.zeros((self.cfg.rows, self.cfg.cols), dtype=bool)
        # V8-B1.6 — générer biome_map AVANT le placement des agents
        # (sinon le pré-calcul affinity_tiles voit l'ancien biome_map)
        if self.cfg.biomes.enabled:
            self._biome_map = generate_biome_map(
                self.cfg.rows, self.cfg.cols, self.cfg.biomes,
                seed=(seed or 0),
            )
        else:
            self._biome_map = np.zeros(
                (self.cfg.rows, self.cfg.cols), dtype=np.int8,
            )
        positions = self._sample_unique_positions(self.cfg.n_agents)
        self._agents = [
            _AgentState(
                agent_id=i, pos=positions[i], energy=self.cfg.start_energy,
                alive=True, parent_id=None, birth_tick=0, generation=0,
                last_repro_tick=-10**9,
            )
            for i in range(self.cfg.n_agents)
        ]
        # V5.2 — root_ancestor_id = own id pour les initiaux
        # V6.1 — initial seeds pour démarrer le cycle agricole
        # V7  — traits initiaux gaussiens si traits.enabled
        init_seeds = (
            self.cfg.planting.initial_seeds
            if self.cfg.planting.enabled else 0
        )
        tcfg = self.cfg.traits
        bcfg = self.cfg.biomes
        # V8-B1.6 — pré-calcul des tiles par biome pour placement
        # des fondateurs dans leur affinity (évite extinction immédiate)
        affinity_tiles: dict[int, list[tuple[int, int]]] = {}
        if bcfg.enabled and bcfg.affinity_enabled:
            for r in range(self.cfg.rows):
                for c in range(self.cfg.cols):
                    b = int(self._biome_map[r, c])
                    if 0 <= b < 4:  # exclut WATER
                        affinity_tiles.setdefault(b, []).append((r, c))
        used_positions: set[tuple[int, int]] = set()
        for a in self._agents:
            a.root_ancestor_id = a.agent_id
            a.seeds = init_seeds
            if tcfg.enabled:
                a.traits = AgentTraits.random(self._spawn_rng, tcfg)
            # V8-B1.6 — tirage affinity uniforme ∈ {0..3} pour fondateurs
            if bcfg.enabled and bcfg.affinity_enabled:
                # V8-C3 C2 — round-robin sur n_initial_affinities (défaut 4).
                # 1 = mono-affinité (tous biome 0), 4 = multi équilibré.
                a.biome_affinity = a.agent_id % bcfg.n_initial_affinities
                # Réassigner sa position dans un tile de son biome (si possible)
                tiles = affinity_tiles.get(a.biome_affinity, [])
                # Filtrer ceux déjà utilisés
                free_tiles = [t for t in tiles if t not in used_positions]
                if free_tiles:
                    idx = int(self._placement_rng.integers(0, len(free_tiles)))
                    a.pos = free_tiles[idx]
                    used_positions.add(a.pos)
                else:
                    used_positions.add(a.pos)
        self._next_agent_id = self.cfg.n_agents
        self._lineage = []
        self._births_last_step = []
        self._nests = {}
        self._builds_last_step = []
        self._nest_visits_total = 0
        self._rest_energy_gained_total = 0.0
        self._family_nest_visits_total = 0
        # V5.3 reset
        self._nest_food_stock = {}
        self._cache_deposits_total = 0
        self._cache_withdrawals_total = 0
        self._cache_energy_deposited_total = 0.0
        self._cache_energy_withdrawn_total = 0.0
        # V6 reset
        self._plants = {}
        self._plants_planted_total = 0
        self._plants_matured_total = 0
        self._last_plant_tick = {}
        # V8-C3 reset
        self._gather_spots = {}
        self._gather_successes_total = 0
        self._gather_failures_total = 0
        self._coop_metrics = CooperativeMetricsTracker(
            CooperativeMetricsConfig()
        )
        # NOTE V8-B1.6 : biome_map est généré au-dessus, avant placement
        # des agents, pour permettre placement par affinity.
        self._initial_food_layout()
        self._refresh_temperature()
        return (
            {a.agent_id: self._observation_for(a.agent_id) for a in self._agents},
            {a.agent_id: {"step": 0, "season": int(self.season)} for a in self._agents},
        )

    def step(
        self, actions: dict[int, int]
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[int, dict[str, Any]],
    ]:
        rewards: dict[int, float] = {}
        terminated: dict[int, bool] = {}
        truncated: dict[int, bool] = {}
        infos: dict[int, dict[str, Any]] = {}
        ate_counts: dict[int, bool] = {}

        # V6.3 — step_count incrémenté en début + construction/plantation
        # AVANT le mouvement (sur position courante). L'agent peut donc agir
        # sur sa cellule actuelle avant de bouger.
        self._step_count += 1
        self._builds_last_step = []
        # V8-B2.0 — reset tokens_this_tick (collect des vocalize)
        self._tokens_this_tick: dict[int, int] = {}
        if self.cfg.build.enabled:
            self._try_constructions()
        if self.cfg.planting.enabled:
            self._try_plantings()
        # V8-C3 — gather_spots : spawn nouveaux + expire vieux
        if self.cfg.cooperative.enabled:
            self._spawn_gather_spots()
            self._decay_gather_spots()
            # Prune vocalize log toutes les 50 ticks (borne mémoire)
            if self._step_count % 50 == 0:
                self._coop_metrics.prune_old_vocalize(self._step_count)

        for agent in self._agents:
            if not agent.alive or agent.agent_id not in actions:
                continue
            action_id = int(actions[agent.agent_id])
            # V8-C3 — action gather_collective = action_id == (4 + n_tokens)
            vcfg = self.cfg.vocabulary
            ccfg = self.cfg.cooperative
            n_vocab = vcfg.n_tokens if vcfg.enabled else 0
            gather_action_id = 4 + n_vocab
            is_gather = (
                ccfg.enabled and action_id == gather_action_id
            )
            is_vocalize = (
                vcfg.enabled and 4 <= action_id < 4 + n_vocab
                and not is_gather
            )

            if is_gather:
                # V8-C3 — tente d'exploiter un gather_spot
                self._try_gather_collective(agent)
                dr, dc = 0, 0
                new_r, new_c = agent.pos
            elif is_vocalize:
                # V8-B2.0 — actions ≥ 4 sont des vocalize (pas mouvement)
                token_id = action_id - 4
                # V8-B2.3 — ablation interventionnelle : vocalize devient
                # no-op après le seuil (pas d'émission, pas de coût)
                ablation_active = (
                    vcfg.disable_vocalize_after_tick is not None
                    and self._step_count > vcfg.disable_vocalize_after_tick
                )
                if 0 <= token_id < vcfg.n_tokens and not ablation_active:
                    self._tokens_this_tick[agent.agent_id] = token_id
                    agent.energy -= vcfg.vocalize_energy_cost
                    # V8-C3 — observer télémétrie
                    if self.cfg.cooperative.enabled:
                        self._coop_metrics.track_vocalize(
                            tick=self._step_count,
                            agent_id=agent.agent_id,
                            token_id=token_id,
                            pos=agent.pos,
                        )
                # Vocalize ne fait pas bouger ; l'agent reste sur place
                # mais subit le metabolism du tile courant
                dr, dc = 0, 0
                new_r, new_c = agent.pos
            else:
                action = Action(action_id if action_id < 4 else 0)
                dr, dc = _DELTAS[action]
                r, c = agent.pos
                new_r = clamp_pos(r, dr, self.cfg.rows)
                new_c = clamp_pos(c, dc, self.cfg.cols)
                agent.pos = (new_r, new_c)

            # Metabolism modulé par la température locale ET le biome (V8-B1.5)
            local_temp = float(self._temp_field[new_r, new_c])
            tile_biome_id = int(self._biome_map[new_r, new_c])
            biome_p = biome_params_for(tile_biome_id, self.cfg.biomes)
            local_metabolism = self.cfg.metabolism * biome_p.metabolism_factor
            local_food_value = self.cfg.food_value * biome_p.food_value_factor
            # V8-B1.6 — bonus/malus selon affinity héritée
            bcfg = self.cfg.biomes
            in_affinity = True
            if (
                bcfg.enabled and bcfg.affinity_enabled
                and agent.biome_affinity is not None
            ):
                if tile_biome_id == agent.biome_affinity:
                    local_metabolism *= bcfg.in_affinity_metabolism
                    local_food_value *= bcfg.in_affinity_food_value
                else:
                    local_metabolism *= bcfg.out_affinity_metabolism
                    local_food_value *= bcfg.out_affinity_food_value
                    in_affinity = False
            # V8-B1.6 — movement_cost amplifié hors affinity (appliqué comme metabolism×)
            move_cost = biome_p.movement_cost
            if not in_affinity:
                move_cost *= bcfg.out_affinity_movement_mult
            # On applique move_cost comme un multiplicateur du metabolism
            # uniquement quand le mouvement a effectivement eu lieu
            if (dr, dc) != (0, 0):
                local_metabolism *= move_cost
            if local_temp < self.cfg.seasonal.cold_threshold:
                local_metabolism *= self.cfg.seasonal.cold_metabolism_factor
            # V8-B1.5 — compétition locale : metabolism augmente avec
            # la densité d'agents voisins dans `radius` cells (Manhattan)
            if self.cfg.competition.enabled:
                ccfg = self.cfg.competition
                n_neigh = 0
                for other in self._agents:
                    if not other.alive or other.agent_id == agent.agent_id:
                        continue
                    odr = abs(other.pos[0] - new_r)
                    odc = abs(other.pos[1] - new_c)
                    if odr + odc <= ccfg.radius:
                        n_neigh += 1
                local_metabolism *= crowd_metabolism_factor(n_neigh, ccfg)

            ate = bool(self._food_mask[new_r, new_c])
            if ate:
                self._food_mask[new_r, new_c] = False
                agent.energy = energy_with_food(
                    agent.energy, local_metabolism,
                    local_food_value, self.cfg.max_energy,
                )
                # V6.1 — manger food → +seeds
                if self.cfg.planting.enabled:
                    agent.seeds += self.cfg.planting.seeds_per_food_eaten
            else:
                agent.energy = energy_no_food(agent.energy, local_metabolism)
            ate_counts[agent.agent_id] = ate

            # V5/V5.2 — rest bonus (propre nid OU nid familial)
            if self.cfg.build.enabled:
                rest_applied = False
                own_nest = self._nests.get(agent.agent_id)
                if own_nest is not None and own_nest.pos == agent.pos:
                    rest_applied = True
                elif self.cfg.build.family_inheritance:
                    for nest in self._nests.values():
                        if nest.pos != agent.pos:
                            continue
                        try:
                            owner_root = self.agent_state(nest.owner_id).root_ancestor_id
                        except KeyError:
                            owner_root = nest.owner_id
                        if owner_root == agent.root_ancestor_id:
                            rest_applied = True
                            break
                if rest_applied:
                    e_before = agent.energy
                    agent.energy = rest_energy_gain(
                        agent.energy, self.cfg.build.rest_bonus, self.cfg.max_energy
                    )
                    self._nest_visits_total += 1
                    self._family_nest_visits_total += 1
                    self._rest_energy_gained_total += agent.energy - e_before

                    # V5.3 — cache deposit / withdrawal (env saisonnier)
                    if self.cfg.cache.enabled:
                        nest_owner_id = self._accessible_nest_owner(agent)
                        if nest_owner_id is not None:
                            ccfg = self.cfg.cache
                            current_cache = self._nest_food_stock.get(nest_owner_id, 0.0)
                            if (
                                agent.energy < ccfg.withdrawal_threshold
                                and current_cache > 0
                            ):
                                effective = min(ccfg.withdrawal_amount, current_cache)
                                new_cache = cache_after_withdrawal(current_cache, effective)
                                actual_amount = current_cache - new_cache
                                if actual_amount > 0:
                                    new_energy = energy_after_withdrawal(
                                        agent.energy, actual_amount, self.cfg.max_energy
                                    )
                                    energy_gained = new_energy - agent.energy
                                    agent.energy = new_energy
                                    self._nest_food_stock[nest_owner_id] = new_cache
                                    self._cache_withdrawals_total += 1
                                    self._cache_energy_withdrawn_total += energy_gained
                            elif (
                                agent.energy >= ccfg.deposit_threshold
                                and current_cache < ccfg.max_capacity
                            ):
                                available = min(
                                    ccfg.deposit_amount, agent.energy - 1.0
                                )
                                new_cache = cache_after_deposit(
                                    current_cache, available, ccfg.max_capacity
                                )
                                actual_deposit = new_cache - current_cache
                                if actual_deposit > 0:
                                    agent.energy -= actual_deposit
                                    self._nest_food_stock[nest_owner_id] = new_cache
                                    self._cache_deposits_total += 1
                                    self._cache_energy_deposited_total += actual_deposit

            r_val = step_reward(local_metabolism, self.cfg.food_value, ate)
            if is_terminated(agent.energy):
                agent.alive = False
                r_val -= self.cfg.death_penalty
                terminated[agent.agent_id] = True
                truncated[agent.agent_id] = False
                if agent.agent_id in self._nests:
                    should_remove = False
                    if self.cfg.build.family_inheritance:
                        if not self.has_living_descendant(agent.root_ancestor_id):
                            should_remove = True
                    else:
                        should_remove = True
                    if should_remove:
                        del self._nests[agent.agent_id]
                        self._nest_food_stock.pop(agent.agent_id, None)
            else:
                terminated[agent.agent_id] = False
            rewards[agent.agent_id] = float(r_val)

        for agent_id in actions:
            if agent_id in terminated and not terminated[agent_id]:
                truncated[agent_id] = self._step_count >= self.cfg.max_steps

        # V4 — reproduction automatique (saisonnier, après mouvement car requiert
        # cellule adjacente libre)
        self._births_last_step = []
        if self.cfg.reproduction.enabled:
            self._try_reproductions()

        # V6 — croissance des plantes
        if self.cfg.planting.enabled:
            self._update_plant_growth()

        # Refresh temperature pour le tick courant
        self._refresh_temperature()
        # Food respawn modulé par la saison (I11)
        self._respawn_food_seasonal()

        obs_dict: dict[int, np.ndarray] = {}
        for agent_id in actions:
            if agent_id not in rewards:
                continue
            obs_dict[agent_id] = self._observation_for(agent_id)
            ar, ac = self._agents[agent_id].pos
            infos[agent_id] = {
                "step": self._step_count,
                "ate": ate_counts.get(agent_id, False),
                "food_count": self.food_count,
                "energy": self._agents[agent_id].energy,
                "alive": self._agents[agent_id].alive,
                "season": int(self.season),
                "local_temp": float(self._temp_field[ar, ac]),
            }
        return obs_dict, rewards, terminated, truncated, infos

    # ─── V8-C3 : gather_collective ─────────────────────────────────────

    def _spawn_gather_spots(self) -> None:
        """V8-C3 — Spawn de gather_spots aléatoires (Poisson)."""
        ccfg = self.cfg.cooperative
        if not ccfg.enabled:
            return
        if len(self._gather_spots) >= ccfg.max_active_spots:
            return
        n_new = int(self._spawn_rng.poisson(ccfg.spawn_lambda))
        if n_new <= 0:
            return
        # Tirer N positions libres (pas de food, pas d'agent, pas de nid)
        occupied = {a.pos for a in self._agents if a.alive}
        occupied |= self.nest_positions
        occupied |= set(self._gather_spots.keys())
        # Tirage aléatoire
        for _ in range(min(n_new, ccfg.max_active_spots - len(self._gather_spots))):
            r = int(self._spawn_rng.integers(0, self.cfg.rows))
            c = int(self._spawn_rng.integers(0, self.cfg.cols))
            if (r, c) in occupied:
                continue
            if self._food_mask[r, c]:
                continue
            spot = GatherSpot(
                pos=(r, c),
                spawned_tick=self._step_count,
                expires_at=self._step_count + ccfg.decay_ticks,
            )
            self._gather_spots[(r, c)] = spot

    def _decay_gather_spots(self) -> None:
        """V8-C3 — Retire les spots expirés."""
        expired = [
            pos for pos, spot in self._gather_spots.items()
            if spot.is_expired(self._step_count)
        ]
        for pos in expired:
            del self._gather_spots[pos]

    def _try_gather_collective(self, agent: _AgentState) -> None:
        """V8-C3 — Si l'agent est sur un spot avec ≥1 partenaire adjacent,
        succès collectif : tous reçoivent +bonus_energy, spot consommé."""
        ccfg = self.cfg.cooperative
        if not ccfg.enabled:
            return
        if agent.pos not in self._gather_spots:
            self._gather_failures_total += 1
            return
        # Compter voisins adjacents (Manhattan ≤ 1, donc 4 directions max)
        ar, ac = agent.pos
        partners = []
        for other in self._agents:
            if not other.alive or other.agent_id == agent.agent_id:
                continue
            d = abs(other.pos[0] - ar) + abs(other.pos[1] - ac)
            if d <= 1:
                partners.append(other)
        if len(partners) < ccfg.min_partners_adjacent:
            self._gather_failures_total += 1
            return
        # Succès collectif : distribuer bonus
        agent.energy = min(
            agent.energy + ccfg.bonus_energy, self.cfg.max_energy,
        )
        for p in partners:
            p.energy = min(
                p.energy + ccfg.bonus_energy, self.cfg.max_energy,
            )
        # Spot consommé
        del self._gather_spots[agent.pos]
        self._gather_successes_total += 1
        # V8-C3 — observer télémétrie 4 métriques
        participants_ids = [agent.agent_id] + [p.agent_id for p in partners]
        alive_positions = [a.pos for a in self._agents if a.alive]
        self._coop_metrics.track_success(
            tick=self._step_count,
            pos=agent.pos,
            participants=participants_ids,
            all_alive_positions=alive_positions,
        )

    def _try_reproductions(self) -> None:
        """V4 — reproduction auto pour env saisonnier (idem MultiAgentFoodGrid).

        V8-B1.6 : gating biome-locked. Si affinity_enabled, parent doit
        être sur un tile de son biome_affinity pour pouvoir se reproduire.
        """
        rcfg = self.cfg.reproduction
        bcfg = self.cfg.biomes
        candidates = sorted(
            (a for a in self._agents if a.alive), key=lambda a: a.agent_id
        )
        for parent in candidates:
            if self.n_alive >= rcfg.max_population:
                break
            if parent.energy < rcfg.energy_threshold:
                continue
            if (self._step_count - parent.last_repro_tick) < rcfg.cooldown_ticks:
                continue
            # V8-B1.6 — gating biome-locked
            required_biome: int | None = None
            if (
                bcfg.enabled and bcfg.affinity_enabled
                and bcfg.reproduction_locked_to_affinity
                and parent.biome_affinity is not None
            ):
                parent_tile = int(self._biome_map[parent.pos[0], parent.pos[1]])
                if parent_tile != parent.biome_affinity:
                    continue  # parent hors son biome → repro impossible
                # Enfant doit aussi naître sur un tile de l'affinity
                required_biome = parent.biome_affinity
            adj = self._find_free_adjacent(parent.pos, required_biome=required_biome)
            if adj is None:
                continue
            # V7 — héritage de traits avec mutation gaussienne
            child_traits = None
            tcfg = self.cfg.traits
            if tcfg.enabled:
                parent_traits = parent.traits or AgentTraits.random(
                    self._spawn_rng, tcfg
                )
                child_traits = parent_traits.mutate(self._spawn_rng, tcfg)
            child = _AgentState(
                agent_id=self._next_agent_id,
                pos=adj,
                energy=rcfg.energy_cost,
                alive=True,
                parent_id=parent.agent_id,
                birth_tick=child_birth_tick(parent.birth_tick, self._step_count),
                generation=child_generation(parent.generation),
                last_repro_tick=-10**9,
                root_ancestor_id=parent.root_ancestor_id,   # V5.2 héritage
                traits=child_traits,                         # V7 héritage+mutation
                biome_affinity=parent.biome_affinity,        # V8-B1.6 héritage 1:1
            )
            parent.energy -= rcfg.energy_cost
            parent.last_repro_tick = self._step_count
            self._agents.append(child)
            self._next_agent_id += 1
            self._births_last_step.append(child.agent_id)
            self._lineage.append(
                LineageEdge(
                    parent_id=parent.agent_id,
                    child_id=child.agent_id,
                    birth_tick=self._step_count,
                    parent_generation=parent.generation,
                    child_generation=child.generation,
                )
            )

    def _try_plantings(self) -> None:
        """V6 — chaque agent éligible plante une graine sur sa cellule.

        V6.1 : consomme `seeds_required` graines de l'agent.
        """
        pcfg = self.cfg.planting
        for agent in self._agents:
            if not self.can_plant_at(agent):
                continue
            agent.energy -= pcfg.energy_cost
            agent.seeds -= pcfg.seeds_required
            self._last_plant_tick[agent.agent_id] = self._step_count
            self._plants[agent.pos] = PlantRecord(
                planter_id=agent.agent_id,
                pos=agent.pos,
                planted_tick=self._step_count,
                matures_at_tick=self._step_count + pcfg.growth_ticks,
            )
            self._plants_planted_total += 1

    def _update_plant_growth(self) -> None:
        """V6 — convertit les plantes mûres en food cells."""
        matured = []
        for pos, plant in self._plants.items():
            if plant.is_mature(self._step_count):
                matured.append(pos)
        for pos in matured:
            r, c = pos
            self._food_mask[r, c] = True
            del self._plants[pos]
            self._plants_matured_total += 1

    def _try_constructions(self) -> None:
        """V5 — construction auto pour env saisonnier (idem MultiAgentFoodGrid)."""
        bcfg = self.cfg.build
        nest_pos_set = self.nest_positions
        for agent in self._agents:
            if not agent.alive:
                continue
            if agent.agent_id in self._nests:
                continue
            if agent.energy < bcfg.energy_threshold:
                continue
            if (self._step_count - agent.last_build_tick) < bcfg.cooldown_ticks:
                continue
            r, c = agent.pos
            if self._food_mask[r, c]:
                continue
            if agent.pos in nest_pos_set:
                continue
            agent.energy = energy_after_build(agent.energy, bcfg.build_cost)
            agent.last_build_tick = self._step_count
            nest = NestRecord(
                owner_id=agent.agent_id, pos=agent.pos, built_tick=self._step_count
            )
            self._nests[agent.agent_id] = nest
            self._builds_last_step.append(nest)
            nest_pos_set.add(agent.pos)

    def spawn_founder(self, affinity: int) -> int | None:
        """V8-B1.7 — Spawn un nouveau fondateur dans un biome donné.

        Cherche un tile libre du biome `affinity` et crée un nouvel agent
        avec une nouvelle root_ancestor_id (nouvelle lignée). Retourne
        l'agent_id du nouveau né, ou None si aucun tile libre.

        Le brain n'est PAS créé ici (responsabilité de l'orchestrateur
        LineageAgent qui peut le hérérer de la seed bank).
        """
        if not self.cfg.biomes.enabled or not self.cfg.biomes.affinity_enabled:
            return None
        # Cherche un tile du biome cible non occupé par un agent vivant
        occupied = {a.pos for a in self._agents if a.alive}
        candidates: list[tuple[int, int]] = []
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                if int(self._biome_map[r, c]) != affinity:
                    continue
                if (r, c) in occupied:
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        idx = int(self._placement_rng.integers(0, len(candidates)))
        pos = candidates[idx]
        new_id = self._next_agent_id
        self._next_agent_id += 1
        new_agent = _AgentState(
            agent_id=new_id, pos=pos,
            energy=self.cfg.biomes.respawn_initial_energy,
            alive=True, parent_id=None,
            birth_tick=self._step_count, generation=0,
            last_repro_tick=-10**9,
            root_ancestor_id=new_id,  # nouvelle lignée
            biome_affinity=affinity,
        )
        self._agents.append(new_agent)
        self._births_last_step.append(new_id)
        return new_id

    def _find_free_adjacent(
        self,
        pos: tuple[int, int],
        required_biome: int | None = None,
    ) -> tuple[int, int] | None:
        """Cherche un tile adjacent libre.

        V8-B1.6 : si `required_biome` est passé, le tile retourné doit
        être du biome demandé (utilisé pour reproduction biome-locked,
        empêche l'enfant de naître dans un biome hostile).
        """
        r, c = pos
        agent_positions = {a.pos for a in self._agents if a.alive}
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.cfg.rows and 0 <= nc < self.cfg.cols):
                continue
            if (nr, nc) in agent_positions:
                continue
            if required_biome is not None:
                if int(self._biome_map[nr, nc]) != required_biome:
                    continue
            return (nr, nc)
        return None

    def _sample_unique_positions(self, n: int) -> list[tuple[int, int]]:
        n_cells = self.cfg.rows * self.cfg.cols
        indices = self._placement_rng.choice(n_cells, size=n, replace=False)
        return [(int(i // self.cfg.cols), int(i % self.cfg.cols)) for i in indices]

    def _initial_food_layout(self) -> None:
        n_cells = self.cfg.rows * self.cfg.cols
        target = int(round(self.cfg.initial_food_density * n_cells))
        if target == 0:
            return
        occupied = {a.pos for a in self._agents}
        available = [
            i for i in range(n_cells)
            if (i // self.cfg.cols, i % self.cfg.cols) not in occupied
        ]
        if not available:
            return
        chosen = self._env_rng.choice(
            available, size=min(target, len(available)), replace=False
        )
        for idx in chosen:
            self._food_mask[int(idx) // self.cfg.cols, int(idx) % self.cfg.cols] = True

    def _respawn_food_seasonal(self) -> None:
        """Food regen modulé par la saison (I11) ET par le biome (V8-B1.5).

        V8-B1.5 : la probabilité qu'une tile reçoive un food spawn est
        pondérée par `biome.food_lambda_factor`. Forêt = abondance,
        désert = rare, tundra = très rare. Tirage pondéré sans remise.
        """
        if self.cfg.food_respawn_lambda <= 0:
            return
        season_factor = get_seasonal_factor(self.season, self.cfg.seasonal)
        effective_lambda = seasonal_lambda(self.cfg.food_respawn_lambda, season_factor)
        if effective_lambda <= 0:
            return
        n_spawn = int(self._spawn_rng.poisson(effective_lambda))
        if n_spawn == 0:
            return
        free = []
        weights = []
        agent_positions = {a.pos for a in self._agents if a.alive}
        nest_pos = self.nest_positions
        plant_pos = set(self._plants.keys())
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                if (
                    not self._food_mask[r, c]
                    and (r, c) not in agent_positions
                    and (r, c) not in nest_pos
                    and (r, c) not in plant_pos
                ):
                    free.append((r, c))
                    # V8-B1.5 — poids = food_lambda_factor du biome
                    biome_p = biome_params_for(
                        int(self._biome_map[r, c]), self.cfg.biomes,
                    )
                    weights.append(biome_p.food_lambda_factor)
        if not free:
            return
        n_actual = min(n_spawn, len(free))
        weights_arr = np.array(weights, dtype=np.float64)
        wsum = weights_arr.sum()
        if wsum <= 0:
            return
        weights_arr /= wsum
        chosen = self._spawn_rng.choice(
            len(free), size=n_actual, replace=False, p=weights_arr,
        )
        for c_idx in chosen:
            r, c = free[int(c_idx)]
            self._food_mask[r, c] = True

    def _observation_for(self, agent_id: int) -> np.ndarray:
        """Observation = pos_self + pos_others + food + [energy, phase, local_temp_norm]."""
        n_cells = self.cfg.rows * self.cfg.cols
        obs = np.zeros(3 * n_cells + 3, dtype=np.float32)
        agent = self._agents[agent_id]
        if agent.alive:
            sr, sc = agent.pos
            obs[sr * self.cfg.cols + sc] = 1.0
        for other in self._agents:
            if other.agent_id == agent_id or not other.alive:
                continue
            r, c = other.pos
            obs[n_cells + r * self.cfg.cols + c] = 1.0
        obs[2 * n_cells : 3 * n_cells] = self._food_mask.flatten().astype(np.float32)
        obs[3 * n_cells] = float(agent.energy) / float(self.cfg.max_energy)
        obs[3 * n_cells + 1] = float(self.phase)
        if agent.alive:
            ar, ac = agent.pos
            local_temp = float(self._temp_field[ar, ac])
        else:
            local_temp = (self.cfg.seasonal.temp_min + self.cfg.seasonal.temp_max) / 2.0
        # Normalize temp dans [0, 1]
        span = self.cfg.seasonal.temp_max - self.cfg.seasonal.temp_min
        obs[3 * n_cells + 2] = (
            (local_temp - self.cfg.seasonal.temp_min) / span if span > 0 else 0.5
        )
        return obs

    def observation_2d_for(self, agent_id: int) -> np.ndarray:
        """Observation 2D (4, R, C) pour ConvDQN V2-W.

        Canaux :
            0 : self_position (1.0 sur la cellule de l'agent)
            1 : others_positions (1.0 par cellule occupée par autre agent vivant)
            2 : food_mask (1.0 si food)
            3 : temperature_field normalisée [0, 1]
        """
        R, C = self.cfg.rows, self.cfg.cols
        obs = np.zeros((4, R, C), dtype=np.float32)
        agent = self._agents[agent_id]
        if agent.alive:
            sr, sc = agent.pos
            obs[0, sr, sc] = 1.0
        for other in self._agents:
            if other.agent_id == agent_id or not other.alive:
                continue
            r, c = other.pos
            obs[1, r, c] = 1.0
        obs[2] = self._food_mask.astype(np.float32)
        # Normalize temperature
        span = self.cfg.seasonal.temp_max - self.cfg.seasonal.temp_min
        if span > 0:
            obs[3] = (self._temp_field - self.cfg.seasonal.temp_min) / span
        else:
            obs[3] = 0.5
        return obs

    @property
    def obs_2d_shape(self) -> tuple[int, int, int]:
        """Shape (channels, rows, cols) attendue par ConvDQNAgent."""
        return (4, self.cfg.rows, self.cfg.cols)

    def observation_2d_dict(self) -> dict[int, np.ndarray]:
        """Retourne un dict {agent_id: obs_2d (4, R, C)} pour tous les agents vivants."""
        return {
            a.agent_id: self.observation_2d_for(a.agent_id)
            for a in self._agents if a.alive
        }
