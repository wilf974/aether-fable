"""MultiAgentFoodGrid — environnement V2 multi-agent indépendant.

API PettingZoo-style :
    reset(seed) → (obs_dict, info_dict)
    step(actions: dict[int, int]) → (obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict)

Conventions :
    - `agent_id` est un int stable assigné à la création (0..n_agents-1).
    - Plusieurs agents peuvent occuper la même cellule.
    - Au tick d'eat, l'agent au `agent_id` le plus bas mange en premier (ordre déterministe).
    - Un agent mort n'apparaît plus dans les dicts retournés.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aetherlife.guardrails.invariants import (
    cache_after_deposit,
    cache_after_withdrawal,
    child_birth_tick,
    child_generation,
    clamp_pos,
    energy_after_build,
    energy_after_withdrawal,
    energy_no_food,
    energy_with_food,
    is_terminated,
    rest_energy_gain,
    step_reward,
)
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig, NestRecord
from aetherlife.world.food_grid import Action, _DELTAS
from aetherlife.world.reproduction import LineageEdge, ReproductionConfig


@dataclass(frozen=True)
class MultiAgentForagerConfig:
    """Configuration V2 multi-agent + reproduction optionnelle V4."""

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
    reproduction: ReproductionConfig = ReproductionConfig()  # V4 default disabled
    build: BuildConfig = BuildConfig()                         # V5 default disabled
    cache: CacheConfig = CacheConfig()                         # V5.3 default disabled

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"rows et cols doivent être > 0")
        if self.n_agents <= 0:
            raise ValueError(f"n_agents doit être > 0 (got {self.n_agents})")
        if self.n_agents > self.rows * self.cols:
            raise ValueError(
                f"n_agents ({self.n_agents}) > rows*cols ({self.rows * self.cols})"
            )
        if self.max_energy <= 0:
            raise ValueError(f"max_energy doit être > 0")
        if not (0 < self.start_energy <= self.max_energy):
            raise ValueError(
                f"start_energy doit être dans (0, max_energy={self.max_energy}]"
            )
        if self.metabolism <= 0:
            raise ValueError(f"metabolism doit être > 0")
        if self.food_value <= 0:
            raise ValueError(f"food_value doit être > 0")
        if self.death_penalty < 0:
            raise ValueError(f"death_penalty doit être >= 0")
        if not (0.0 <= self.initial_food_density <= 1.0):
            raise ValueError(f"initial_food_density doit être dans [0, 1]")
        if self.food_respawn_lambda < 0:
            raise ValueError(f"food_respawn_lambda doit être >= 0")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps doit être > 0")

    @property
    def obs_dim(self) -> int:
        """Dim per-agent = self_one_hot + others_one_hot + food_one_hot + my_energy."""
        return 3 * self.rows * self.cols + 1


@dataclass
class _AgentState:
    """État d'un agent. V4 ajoute lineage (parent_id, birth_tick, generation)."""

    agent_id: int
    pos: tuple[int, int]
    energy: float
    alive: bool = True
    # V4 lineage
    parent_id: int | None = None
    birth_tick: int = 0
    generation: int = 0
    last_repro_tick: int = -10**9   # permet la 1ère reproduction immédiate
    # V5 — construction
    last_build_tick: int = -10**9
    # V5.2 — lineage root pour héritage familial (default = own id)
    root_ancestor_id: int = -1
    # V6.1 — graines accumulées en mangeant food, dépensées en plantant
    seeds: int = 0


class MultiAgentFoodGrid:
    """Multi-agent grid avec food et énergie. State-machine pure Python+numpy."""

    def __init__(self, cfg: MultiAgentForagerConfig | None = None) -> None:
        self.cfg = cfg or MultiAgentForagerConfig()
        self._food_mask: np.ndarray = np.zeros(
            (self.cfg.rows, self.cfg.cols), dtype=bool
        )
        self._agents: list[_AgentState] = []
        self._step_count: int = 0
        self._env_rng: np.random.Generator = np.random.default_rng()
        self._spawn_rng: np.random.Generator = np.random.default_rng()
        self._placement_rng: np.random.Generator = np.random.default_rng()
        # V4 lineage tracking
        self._lineage: list[LineageEdge] = []
        self._next_agent_id: int = 0
        self._births_last_step: list[int] = []  # agent_ids nés au step courant
        # V5 — construction
        self._nests: dict[int, NestRecord] = {}      # owner_id → nid
        self._builds_last_step: list[NestRecord] = []
        self._nest_visits_total: int = 0
        self._rest_energy_gained_total: float = 0.0
        # V5.2 — family inheritance metrics
        self._family_nest_visits_total: int = 0
        # V5.3 — cache state + métriques
        self._nest_food_stock: dict[int, float] = {}      # owner_id → cache content
        self._cache_deposits_total: int = 0
        self._cache_withdrawals_total: int = 0
        self._cache_energy_deposited_total: float = 0.0
        self._cache_energy_withdrawn_total: float = 0.0

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
    def food_count(self) -> int:
        return int(self._food_mask.sum())

    @property
    def food_mask(self) -> np.ndarray:
        return self._food_mask.copy()

    @property
    def n_alive(self) -> int:
        return sum(1 for a in self._agents if a.alive)

    @property
    def n_dead(self) -> int:
        return sum(1 for a in self._agents if not a.alive)

    @property
    def alive_agent_ids(self) -> list[int]:
        return [a.agent_id for a in self._agents if a.alive]

    @property
    def lineage(self) -> list[LineageEdge]:
        """Liste des arêtes parent→enfant depuis le dernier reset."""
        return list(self._lineage)

    @property
    def births_last_step(self) -> list[int]:
        """agent_ids nés au step le plus récent (vidé à chaque step)."""
        return list(self._births_last_step)

    @property
    def n_births_total(self) -> int:
        return len(self._lineage)

    # ─── V5 construction state ─────────────────────────────────────────
    @property
    def nests(self) -> dict[int, NestRecord]:
        """Dict owner_id → NestRecord (copie pour éviter mutation externe)."""
        return dict(self._nests)

    @property
    def n_nests(self) -> int:
        return len(self._nests)

    @property
    def builds_last_step(self) -> list[NestRecord]:
        return list(self._builds_last_step)

    @property
    def nest_positions(self) -> set[tuple[int, int]]:
        """Set des cellules occupées par un nid."""
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
        """V5.2 — ticks où un agent était sur un nid de sa lignée (incl. propre)."""
        return self._family_nest_visits_total

    # ─── V5.3 cache state ──────────────────────────────────────────────
    @property
    def nest_food_stock(self) -> dict[int, float]:
        """Dict owner_id → quantité stockée dans le cache."""
        return dict(self._nest_food_stock)

    @property
    def total_cached_food(self) -> float:
        """Somme de tous les caches actifs."""
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

    # ─── V5.2 lineage helpers ──────────────────────────────────────────
    def root_ancestor_of(self, agent_id: int) -> int:
        """Retourne le root_ancestor_id d'un agent (par lookup direct)."""
        return self.agent_state(agent_id).root_ancestor_id

    def lineage_ids(self, root_id: int) -> list[int]:
        """Tous les agent_ids (vivants + morts) appartenant à une lignée."""
        return [
            a.agent_id for a in self._agents if a.root_ancestor_id == root_id
        ]

    def has_living_descendant(self, root_id: int) -> bool:
        """True s'il existe au moins un agent vivant dans cette lignée."""
        return any(
            a.alive and a.root_ancestor_id == root_id for a in self._agents
        )

    def _accessible_nest_owner(self, agent: _AgentState) -> int | None:
        """V5.3 — owner_id du nid sur la cellule de l'agent et accessible.

        Accessible = own nest OU nid familial (si family_inheritance).
        Retourne None si pas de nid accessible sur cette cellule.
        """
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
        # V4 : agent_id n'est plus garanti = index. Lookup explicite.
        for a in self._agents:
            if a.agent_id == agent_id:
                return a
        raise KeyError(f"agent_id={agent_id} not found")

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
        if seed is not None:
            self._env_rng = np.random.default_rng(seed)
            self._spawn_rng = np.random.default_rng(seed + 1)
            self._placement_rng = np.random.default_rng(seed + 2)
        self._step_count = 0
        self._food_mask = np.zeros((self.cfg.rows, self.cfg.cols), dtype=bool)
        positions = self._sample_unique_positions(self.cfg.n_agents)
        self._agents = [
            _AgentState(
                agent_id=i, pos=positions[i], energy=self.cfg.start_energy,
                alive=True, parent_id=None, birth_tick=0, generation=0,
                last_repro_tick=-10**9,
            )
            for i in range(self.cfg.n_agents)
        ]
        # V5.2 — set root_ancestor_id = own id pour les initiaux
        for a in self._agents:
            a.root_ancestor_id = a.agent_id
        self._next_agent_id = self.cfg.n_agents
        self._lineage = []
        self._births_last_step = []
        self._nests = {}
        self._builds_last_step = []
        self._nest_visits_total = 0
        self._rest_energy_gained_total = 0.0
        self._family_nest_visits_total = 0    # V5.2
        # V5.3 reset
        self._nest_food_stock = {}
        self._cache_deposits_total = 0
        self._cache_withdrawals_total = 0
        self._cache_energy_deposited_total = 0.0
        self._cache_energy_withdrawn_total = 0.0
        self._initial_food_layout()
        return (
            {a.agent_id: self._observation_for(a.agent_id) for a in self._agents},
            {a.agent_id: {"step": 0} for a in self._agents},
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

        for agent in self._agents:
            if not agent.alive:
                continue
            if agent.agent_id not in actions:
                continue
            action = Action(int(actions[agent.agent_id]))
            dr, dc = _DELTAS[action]
            r, c = agent.pos
            new_r = clamp_pos(r, dr, self.cfg.rows)
            new_c = clamp_pos(c, dc, self.cfg.cols)
            agent.pos = (new_r, new_c)

            ate = bool(self._food_mask[new_r, new_c])
            if ate:
                self._food_mask[new_r, new_c] = False
                agent.energy = energy_with_food(
                    agent.energy, self.cfg.metabolism,
                    self.cfg.food_value, self.cfg.max_energy,
                )
            else:
                agent.energy = energy_no_food(agent.energy, self.cfg.metabolism)
            ate_counts[agent.agent_id] = ate

            # V5 — rest bonus si l'agent est sur son propre nid (ou nid familial V5.2)
            if self.cfg.build.enabled:
                rest_applied = False
                own_nest = self._nests.get(agent.agent_id)
                if own_nest is not None and own_nest.pos == agent.pos:
                    rest_applied = True
                elif self.cfg.build.family_inheritance:
                    # V5.2 — chercher un nid de la même lignée sur cette cellule
                    for nest in self._nests.values():
                        if nest.pos != agent.pos:
                            continue
                        try:
                            owner_root = self.agent_state(nest.owner_id).root_ancestor_id
                        except KeyError:
                            # owner mort et déjà retiré → on stocke encore le nid par
                            # root_id ; cas géré : nest_owner_root_id séparé (cf. ci-dessous)
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

                    # V5.3 — cache deposit / withdrawal sur le nid (la valeur
                    # `accessible_nest_owner_id` détermine quel cache utiliser)
                    if self.cfg.cache.enabled:
                        nest_owner_id = self._accessible_nest_owner(agent)
                        if nest_owner_id is not None:
                            ccfg = self.cfg.cache
                            current_cache = self._nest_food_stock.get(nest_owner_id, 0.0)
                            # Retrait prioritaire si l'agent a peu d'énergie
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
                            # Sinon dépôt si l'agent est en surplus
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

            r = step_reward(self.cfg.metabolism, self.cfg.food_value, ate)
            if is_terminated(agent.energy):
                agent.alive = False
                r -= self.cfg.death_penalty
                terminated[agent.agent_id] = True
                truncated[agent.agent_id] = False
                # V5/V5.2 — cleanup nid du défunt
                if agent.agent_id in self._nests:
                    should_remove = False
                    if self.cfg.build.family_inheritance:
                        # V5.2 — conserver le nid si un descendant vit (same root)
                        if not self.has_living_descendant(agent.root_ancestor_id):
                            should_remove = True
                    else:
                        should_remove = True
                    if should_remove:
                        del self._nests[agent.agent_id]
                        # V5.3 — cache perdu avec le nid
                        self._nest_food_stock.pop(agent.agent_id, None)
            else:
                terminated[agent.agent_id] = False
            rewards[agent.agent_id] = float(r)

        self._step_count += 1
        # truncation pour ceux qui n'ont pas été terminés
        for agent_id in actions:
            if agent_id in terminated and not terminated[agent_id]:
                trunc = self._step_count >= self.cfg.max_steps
                truncated[agent_id] = trunc

        # V4 — reproduction automatique (après step, avant respawn food)
        self._births_last_step = []
        if self.cfg.reproduction.enabled:
            self._try_reproductions()

        # V5 — construction automatique
        self._builds_last_step = []
        if self.cfg.build.enabled:
            self._try_constructions()

        self._respawn_food()

        obs_dict: dict[int, np.ndarray] = {}
        for agent_id in actions:
            if agent_id not in rewards:
                continue
            obs_dict[agent_id] = self._observation_for(agent_id)
            infos[agent_id] = {
                "step": self._step_count,
                "ate": ate_counts.get(agent_id, False),
                "food_count": self.food_count,
                "energy": self._agents[agent_id].energy,
                "alive": self._agents[agent_id].alive,
            }
        return obs_dict, rewards, terminated, truncated, infos

    def _try_reproductions(self) -> None:
        """V4 — boucle sur agents éligibles et tente de créer des enfants.

        Conditions per agent :
            - alive
            - energy >= reproduction.energy_threshold
            - step_count - last_repro_tick >= cooldown_ticks
            - pop < max_population
            - cellule adjacente libre (pas d'autre agent)

        Effets si succès :
            - parent.energy -= energy_cost
            - parent.last_repro_tick = current step
            - nouvel agent (next_id) avec energy=energy_cost, generation=parent+1
            - LineageEdge ajouté
        """
        rcfg = self.cfg.reproduction
        # Snapshot des agents éligibles (parcours ordre id ascendant)
        candidates = sorted(
            (a for a in self._agents if a.alive),
            key=lambda a: a.agent_id,
        )
        for parent in candidates:
            if self.n_alive >= rcfg.max_population:
                break
            if parent.energy < rcfg.energy_threshold:
                continue
            if (self._step_count - parent.last_repro_tick) < rcfg.cooldown_ticks:
                continue
            adj = self._find_free_adjacent(parent.pos)
            if adj is None:
                continue
            # Naissance
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

    def _try_constructions(self) -> None:
        """V5 — chaque agent vivant éligible tente de construire un nid sur sa cellule.

        Conditions :
            - alive
            - energy ≥ build.energy_threshold
            - step_count - last_build_tick ≥ cooldown_ticks
            - pas déjà de nid pour cet agent (max 1 par I16)
            - pas de food sur la cellule (évite collision avec respawn)
            - pas de nid d'autre agent sur cette cellule
        """
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
            # Build
            agent.energy = energy_after_build(agent.energy, bcfg.build_cost)
            agent.last_build_tick = self._step_count
            nest = NestRecord(
                owner_id=agent.agent_id, pos=agent.pos, built_tick=self._step_count
            )
            self._nests[agent.agent_id] = nest
            self._builds_last_step.append(nest)
            nest_pos_set.add(agent.pos)

    def _find_free_adjacent(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """Cherche une cellule 4-voisine libre (pas d'agent vivant)."""
        r, c = pos
        agent_positions = {a.pos for a in self._agents if a.alive}
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.cfg.rows and 0 <= nc < self.cfg.cols):
                continue
            if (nr, nc) in agent_positions:
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
        available_indices = [
            i
            for i in range(n_cells)
            if (i // self.cfg.cols, i % self.cfg.cols) not in occupied
        ]
        if len(available_indices) == 0:
            return
        chosen = self._env_rng.choice(
            available_indices, size=min(target, len(available_indices)), replace=False
        )
        for idx in chosen:
            self._food_mask[int(idx) // self.cfg.cols, int(idx) % self.cfg.cols] = True

    def _respawn_food(self) -> None:
        if self.cfg.food_respawn_lambda <= 0:
            return
        n_spawn = int(self._spawn_rng.poisson(self.cfg.food_respawn_lambda))
        if n_spawn == 0:
            return
        free = []
        agent_positions = {a.pos for a in self._agents if a.alive}
        nest_pos = self.nest_positions
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                if (
                    not self._food_mask[r, c]
                    and (r, c) not in agent_positions
                    and (r, c) not in nest_pos
                ):
                    free.append((r, c))
        if len(free) == 0:
            return
        n_actual = min(n_spawn, len(free))
        chosen = self._spawn_rng.choice(len(free), size=n_actual, replace=False)
        for c_idx in chosen:
            r, c = free[int(c_idx)]
            self._food_mask[r, c] = True

    def _observation_for(self, agent_id: int) -> np.ndarray:
        """Build observation pour un agent donné."""
        n_cells = self.cfg.rows * self.cfg.cols
        obs = np.zeros(3 * n_cells + 1, dtype=np.float32)
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
        obs[-1] = float(agent.energy) / float(self.cfg.max_energy)
        return obs

    def render_ascii(self) -> str:
        """Rendu texte minimal."""
        agent_at: dict[tuple[int, int], int] = {}
        for a in self._agents:
            if a.alive:
                agent_at[a.pos] = a.agent_id
        lines: list[str] = []
        for r in range(self.cfg.rows):
            row_chars: list[str] = []
            for c in range(self.cfg.cols):
                if (r, c) in agent_at:
                    aid = agent_at[(r, c)]
                    row_chars.append(str(aid % 10))
                elif self._food_mask[r, c]:
                    row_chars.append("*")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        lines.append(
            f"alive={self.n_alive}/{self.cfg.n_agents} dead={self.n_dead} "
            f"step={self._step_count} food={self.food_count}"
        )
        return "\n".join(lines)
