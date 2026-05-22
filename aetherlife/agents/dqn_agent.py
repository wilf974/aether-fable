"""DQNAgent AetherLife — réutilise l'infra MW_IA DQN sur FoodGrid.

Wrapper qui adapte l'interface :
    - act(observation, *, info=None) → int   [interface AetherLife]
    - act(state, *, greedy=False) → int      [interface MW_IA]

Et bridge `observe()` pour pousser dans le replay buffer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from mw_ia.agents.dqn import DQNAgent as _MwIaDQNAgent
from mw_ia.config import DQNConfig

from aetherlife.world.food_grid import FoodGrid


class DQNAgent:
    """Wrapper AetherLife autour de mw_ia.agents.dqn.DQNAgent.

    `FoodGrid` expose déjà `n_states` et `n_actions` (cf. food_grid.py), donc
    `_MwIaDQNAgent` accepte directement notre env (typage runtime non-vérifié
    en Python — seules les propriétés `n_states` / `n_actions` sont lues).
    """

    def __init__(
        self,
        env: FoodGrid,
        cfg: DQNConfig | None = None,
        *,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.env = env
        self.cfg = cfg or DQNConfig()
        self._impl = _MwIaDQNAgent(env, self.cfg, device=device, seed=seed)

    @property
    def epsilon(self) -> float:
        return self._impl.epsilon

    @property
    def global_step(self) -> int:
        return self._impl.global_step

    @property
    def last_loss(self) -> float | None:
        return self._impl.last_loss

    def act(
        self,
        observation: np.ndarray,
        *,
        info: dict | None = None,
        greedy: bool = False,
    ) -> int:
        """Choisit une action. `greedy=True` ignore ε-greedy (pour eval)."""
        return self._impl.act(observation, greedy=greedy)

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        """Push une transition + déclenche train_step + sync target si dû."""
        return self._impl.observe(state, action, reward, next_state, done)

    def save(self, path: str | Path) -> None:
        self._impl.save(path)

    def load(self, path: str | Path) -> None:
        self._impl.load(path)
