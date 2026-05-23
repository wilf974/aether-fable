"""ConvDQNAgent AetherLife — wrap mw_ia V2-W (ConvDQN + Double DQN).

Pour env saisonnier 2D : 4 canaux (self + others + food + temperature).
Double DQN activé par défaut (V2-W de MW_IA, top archi prouvé sur procedural).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from mw_ia.agents.conv_dqn import ConvDQNAgent as _MwIaConvDQNAgent
from mw_ia.config import ConvDQNConfig


class ConvDQNAgent:
    """Wrap `mw_ia.agents.conv_dqn.ConvDQNAgent` (V2-W) sur env AetherLife 2D.

    Args:
        in_channels: nombre de canaux d'entrée (4 pour env saisonnier).
        rows / cols: dimensions spatiales.
        n_actions: nombre d'actions discrètes.
        cfg: ConvDQNConfig MW_IA (defaults V2-W : double_dqn=True).
    """

    def __init__(
        self,
        in_channels: int,
        rows: int,
        cols: int,
        n_actions: int,
        cfg: ConvDQNConfig | None = None,
        *,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or ConvDQNConfig()
        self._impl = _MwIaConvDQNAgent(
            in_channels=in_channels, rows=rows, cols=cols,
            n_actions=n_actions, cfg=self.cfg, device=device, seed=seed,
        )

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
        return self._impl.act(observation, greedy=greedy)

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        return self._impl.observe(state, action, reward, next_state, done)

    def save(self, path: str | Path) -> None:
        self._impl.save(path)

    def load(self, path: str | Path) -> None:
        self._impl.load(path)
