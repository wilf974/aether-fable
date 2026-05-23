"""RecurrentDQNAgent AetherLife — wrap mw_ia DRQN pour SoloSeasonalEnv.

Thin wrapper qui expose une API homogène (act/observe/end_episode/save/load)
adaptée aux runners AetherLife.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent as _MwIaDRQNAgent
from mw_ia.config import DRQNConfig


class RecurrentDQNAgent:
    """Wrap `mw_ia.agents.recurrent_dqn.RecurrentDQNAgent` sur env single-agent AetherLife.

    Args:
        obs_dim: dimension de l'observation (par ex. `env.n_states`).
        n_actions: nombre d'actions discrètes.
        cfg: DRQNConfig MW_IA. Si None, defaults V2-Y.
        device: "cuda" ou "cpu".
        seed: graine RNG.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: DRQNConfig | None = None,
        *,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or DRQNConfig()
        self._impl = _MwIaDRQNAgent(
            obs_dim=obs_dim, n_actions=n_actions, cfg=self.cfg,
            device=device, seed=seed,
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

    def reset_hidden(self) -> None:
        self._impl.reset_hidden()

    def begin_episode(self) -> None:
        self._impl.begin_episode()

    def end_episode(self) -> dict[str, float]:
        return self._impl.end_episode()

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
