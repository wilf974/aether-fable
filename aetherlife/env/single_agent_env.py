"""SoloForagerEnv — wrapper Gymnasium autour de FoodGrid."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aetherlife.config import FoodGridConfig
from aetherlife.world.food_grid import FoodGrid


class SoloForagerEnv(gym.Env):
    """Wrapper Gymnasium-compatible autour de FoodGrid.

    - Observation : Box[0, 1]^(2*rows*cols + 1).
    - Action : Discrete(4).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cfg: FoodGridConfig | None = None) -> None:
        super().__init__()
        self._world = FoodGrid(cfg)
        cfg = self._world.cfg
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(cfg.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._world.n_actions)

    @property
    def world(self) -> FoodGrid:
        return self._world

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return self._world.reset(seed=seed)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._world.step(action)

    def render(self) -> str:
        return self._world.render_ascii()
