"""IndependentConvDQNAgent — IDQN shared-weights avec ConvDQN+Double DQN (V2-W).

Pour V3.7 : multi-agent saisonnier complexe où la perception spatiale 2D
devrait dominer le MLP feedforward de V2 IndependentDQNAgent.

Un seul ConvQNetwork partagé entre les N agents — réutilise l'infra V2 (act_dict
+ observe_dict) avec tensors (4, R, C) au lieu de vecteurs flat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from mw_ia.agents.conv_dqn import ConvDQNAgent as _MwIaConvDQNAgent
from mw_ia.config import ConvDQNConfig

from aetherlife.world.seasonal_grid import SeasonalMultiAgentFoodGrid


class IndependentConvDQNAgent:
    """IDQN shared-weights avec ConvDQN+Double DQN (V2-W MW_IA).

    Args:
        env: SeasonalMultiAgentFoodGrid avec `obs_2d_shape` exposé.
        cfg: ConvDQNConfig (defaults V2-W : double_dqn=True).
        device: "cuda" ou "cpu".
        seed: graine RNG.
    """

    def __init__(
        self,
        env: SeasonalMultiAgentFoodGrid,
        cfg: ConvDQNConfig | None = None,
        *,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.env = env
        self.cfg = cfg or ConvDQNConfig()
        in_ch, R, C = env.obs_2d_shape
        self._impl = _MwIaConvDQNAgent(
            in_channels=in_ch, rows=R, cols=C,
            n_actions=env.n_actions, cfg=self.cfg,
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

    def act_dict(
        self,
        obs_dict: dict[int, np.ndarray],
        *,
        greedy: bool = False,
    ) -> dict[int, int]:
        """Forward chaque obs 2D et retourne un dict d'actions."""
        return {
            agent_id: self._impl.act(obs2d, greedy=greedy)
            for agent_id, obs2d in obs_dict.items()
        }

    def observe_dict(
        self,
        prev_obs: dict[int, np.ndarray],
        actions: dict[int, int],
        rewards: dict[int, float],
        next_obs: dict[int, np.ndarray],
        dones: dict[int, bool],
    ) -> dict[str, float]:
        """Push toutes les transitions dans le replay buffer commun."""
        metrics: dict[str, float] = {}
        for agent_id, obs in prev_obs.items():
            if agent_id not in actions or agent_id not in rewards:
                continue
            done = dones.get(agent_id, False)
            n_obs = next_obs.get(agent_id, obs)
            m = self._impl.observe(obs, actions[agent_id], rewards[agent_id], n_obs, done)
            metrics.update(m)
        return metrics

    def save(self, path: str | Path) -> None:
        self._impl.save(path)

    def load(self, path: str | Path) -> None:
        self._impl.load(path)
