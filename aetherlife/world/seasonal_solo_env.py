"""SoloSeasonalEnv — single-agent saisonnier (V3.5).

Wrap SeasonalMultiAgentFoodGrid avec n_agents=1 et expose l'API single-agent
(comme FoodGrid V1) pour comparer MLP vs LSTM sur env non-stationnaire.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from aetherlife.world.food_grid import Action
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


class SoloSeasonalEnv:
    """API single-agent (reset/step → tuples) sur env saisonnier multi-agent (n_agents=1)."""

    def __init__(
        self,
        cfg: SeasonalMultiAgentConfig | None = None,
        *,
        seasonal: SeasonalConfig | None = None,
    ) -> None:
        if cfg is None:
            cfg = SeasonalMultiAgentConfig(
                n_agents=1,
                seasonal=seasonal or SeasonalConfig(),
            )
        if cfg.n_agents != 1:
            cfg = SeasonalMultiAgentConfig(
                rows=cfg.rows, cols=cfg.cols, n_agents=1,
                max_energy=cfg.max_energy, start_energy=cfg.start_energy,
                metabolism=cfg.metabolism, food_value=cfg.food_value,
                death_penalty=cfg.death_penalty,
                initial_food_density=cfg.initial_food_density,
                food_respawn_lambda=cfg.food_respawn_lambda,
                max_steps=cfg.max_steps,
                seasonal=cfg.seasonal,
            )
        self.cfg = cfg
        self._inner = SeasonalMultiAgentFoodGrid(cfg)

    # --- API MW_IA-compat (n_states + n_actions) ---

    @property
    def n_actions(self) -> int:
        return self._inner.n_actions

    @property
    def n_states(self) -> int:
        return self.cfg.obs_dim

    @property
    def step_count(self) -> int:
        return self._inner.step_count

    @property
    def pos(self) -> tuple[int, int]:
        return self._inner.agent_state(0).pos

    @property
    def energy(self) -> float:
        return self._inner.agent_state(0).energy

    @property
    def food_count(self) -> int:
        return self._inner.food_count

    @property
    def food_mask(self) -> np.ndarray:
        return self._inner.food_mask

    @property
    def temperature_field(self) -> np.ndarray:
        return self._inner.temperature_field

    @property
    def phase(self) -> float:
        return self._inner.phase

    @property
    def season(self) -> int:
        return int(self._inner.season)

    def agent_state(self, agent_id: int = 0):
        return self._inner.agent_state(agent_id)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs_dict, info_dict = self._inner.reset(seed=seed)
        return obs_dict[0], info_dict[0]

    def step(
        self, action: int | Action
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs_dict, rewards, terminated, truncated, infos = self._inner.step(
            {0: int(action)}
        )
        # Si l'agent meurt au step courant, obs_dict[0] est rempli (cf. step impl)
        obs = obs_dict.get(0, np.zeros(self.cfg.obs_dim, dtype=np.float32))
        return (
            obs,
            float(rewards.get(0, 0.0)),
            bool(terminated.get(0, False)),
            bool(truncated.get(0, False)),
            infos.get(0, {}),
        )

    def observation_2d(self) -> np.ndarray:
        """Observation 2D (4, R, C) pour ConvDQN V2-W."""
        return self._inner.observation_2d_for(0)

    @property
    def obs_2d_shape(self) -> tuple[int, int, int]:
        return self._inner.obs_2d_shape
