"""IndependentDQNAgent — IDQN avec shared weights pour V2 multi-agent.

Un seul réseau DQN partagé entre tous les agents. Chaque agent observe son
propre `obs[i]` et choisit indépendamment via le même réseau. Toutes les
transitions sont poussées dans le replay buffer commun.

Avantages :
- VRAM minimale (1 réseau × N inférences)
- Homogénéité d'apprentissage (toutes les expériences renforcent le même réseau)

Compromis V2 : pas de différenciation des agents (V5 ajoutera la communication
et MAPPO pour CTDE).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from mw_ia.agents.dqn import DQNAgent as _MwIaDQNAgent
from mw_ia.config import DQNConfig

from aetherlife.world.multi_agent_grid import MultiAgentFoodGrid


class IndependentDQNAgent:
    """IDQN shared-weights — un seul `_MwIaDQNAgent` partagé.

    Le wrapper expose une API per-agent (act_dict, observe_dict) qui boucle
    sur les agents vivants et appelle l'agent interne en batch séquentiel.
    """

    def __init__(
        self,
        env: MultiAgentFoodGrid,
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

    def act_dict(
        self,
        obs_dict: dict[int, np.ndarray],
        *,
        greedy: bool = False,
    ) -> dict[int, int]:
        """Forward chaque agent et retourne un dict d'actions."""
        return {
            agent_id: self._impl.act(obs, greedy=greedy)
            for agent_id, obs in obs_dict.items()
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
            m = self._impl.observe(
                obs, actions[agent_id], rewards[agent_id], n_obs, done
            )
            metrics.update(m)
        return metrics

    def save(self, path: str | Path) -> None:
        self._impl.save(path)

    def load(self, path: str | Path) -> None:
        self._impl.load(path)
