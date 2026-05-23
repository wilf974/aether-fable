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
    clamp_pos,
    clamp_temp,
    energy_no_food,
    energy_with_food,
    is_terminated,
    season_phase,
    seasonal_lambda,
    step_reward,
)
from aetherlife.world.food_grid import Action, _DELTAS
from aetherlife.world.multi_agent_grid import _AgentState


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
        self._agents: list[_AgentState] = []
        self._step_count: int = 0
        self._env_rng: np.random.Generator = np.random.default_rng()
        self._spawn_rng: np.random.Generator = np.random.default_rng()
        self._placement_rng: np.random.Generator = np.random.default_rng()

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

    def agent_state(self, agent_id: int) -> _AgentState:
        return self._agents[agent_id]

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
        positions = self._sample_unique_positions(self.cfg.n_agents)
        self._agents = [
            _AgentState(
                agent_id=i, pos=positions[i], energy=self.cfg.start_energy, alive=True
            )
            for i in range(self.cfg.n_agents)
        ]
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

        for agent in self._agents:
            if not agent.alive or agent.agent_id not in actions:
                continue
            action = Action(int(actions[agent.agent_id]))
            dr, dc = _DELTAS[action]
            r, c = agent.pos
            new_r = clamp_pos(r, dr, self.cfg.rows)
            new_c = clamp_pos(c, dc, self.cfg.cols)
            agent.pos = (new_r, new_c)

            # Metabolism modulé par la température locale
            local_temp = float(self._temp_field[new_r, new_c])
            local_metabolism = self.cfg.metabolism
            if local_temp < self.cfg.seasonal.cold_threshold:
                local_metabolism *= self.cfg.seasonal.cold_metabolism_factor

            ate = bool(self._food_mask[new_r, new_c])
            if ate:
                self._food_mask[new_r, new_c] = False
                agent.energy = energy_with_food(
                    agent.energy, local_metabolism,
                    self.cfg.food_value, self.cfg.max_energy,
                )
            else:
                agent.energy = energy_no_food(agent.energy, local_metabolism)
            ate_counts[agent.agent_id] = ate

            r_val = step_reward(local_metabolism, self.cfg.food_value, ate)
            if is_terminated(agent.energy):
                agent.alive = False
                r_val -= self.cfg.death_penalty
                terminated[agent.agent_id] = True
                truncated[agent.agent_id] = False
            else:
                terminated[agent.agent_id] = False
            rewards[agent.agent_id] = float(r_val)

        self._step_count += 1
        for agent_id in actions:
            if agent_id in terminated and not terminated[agent_id]:
                truncated[agent_id] = self._step_count >= self.cfg.max_steps

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
        """Food regen modulé par la saison (I11)."""
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
        agent_positions = {a.pos for a in self._agents if a.alive}
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                if not self._food_mask[r, c] and (r, c) not in agent_positions:
                    free.append((r, c))
        if not free:
            return
        n_actual = min(n_spawn, len(free))
        chosen = self._spawn_rng.choice(len(free), size=n_actual, replace=False)
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
