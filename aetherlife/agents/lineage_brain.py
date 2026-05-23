"""V8-B1 — LineageBrain : RL DQN partagé par lignée (root_ancestor_id).

Une instance de `LineageBrain` détient un Q-network DQN partagé entre tous
les agents vivants d'une même lignée. À la reproduction d'un fondateur
(création d'une nouvelle lignée racine), `inherit_from()` clone le cerveau
du parent et applique une mutation gaussienne sur les poids.

Architecture :
    - QNetwork MLP compact (default hidden 64×64)
    - ReplayBuffer partagé par la lignée
    - DQNTrainer (gradient + target sync)
    - Epsilon-greedy avec décroissance linéaire

Réutilise les composants MW_IA (`QNetwork`, `ReplayBuffer`, `DQNTrainer`)
qui sont env-agnostiques.

Voir la spec : `docs/superpowers/specs/2026-05-23-aetherlife-v8-b1-cognitive-inheritance-design.md`
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BrainConfig:
    """V8-B1 — Configuration du cerveau par lignée."""

    enabled: bool = False  # compat V7 et avant

    # Architecture
    hidden_dims: tuple[int, ...] = (64, 64)

    # RL hyperparams (DQN classique, calibration V1.5 piège #6)
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_capacity: int = 50_000
    min_replay_to_learn: int = 500
    train_every: int = 4

    # Exploration
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    target_sync_steps: int = 300

    # Héritage
    mutation_std: float = 0.02   # std du bruit gaussien à l'héritage

    # Observation (égocentrique)
    vision_radius: int = 5       # 11×11 fenêtre

    # Practical
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not (0 < self.lr < 1):
            raise ValueError(f"lr doit être dans (0, 1), got {self.lr}")
        if not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma doit être dans [0, 1], got {self.gamma}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être > 0, got {self.batch_size}")
        if self.buffer_capacity <= 0:
            raise ValueError(
                f"buffer_capacity doit être > 0, got {self.buffer_capacity}"
            )
        if self.min_replay_to_learn < 0:
            raise ValueError(
                f"min_replay_to_learn doit être >= 0, got {self.min_replay_to_learn}"
            )
        if self.train_every <= 0:
            raise ValueError(f"train_every doit être > 0, got {self.train_every}")
        if not (0 <= self.epsilon_start <= 1):
            raise ValueError(
                f"epsilon_start doit être dans [0, 1], got {self.epsilon_start}"
            )
        if not (0 <= self.epsilon_end <= 1):
            raise ValueError(
                f"epsilon_end doit être dans [0, 1], got {self.epsilon_end}"
            )
        if self.epsilon_decay_steps <= 0:
            raise ValueError(
                f"epsilon_decay_steps doit être > 0, got {self.epsilon_decay_steps}"
            )
        if self.target_sync_steps <= 0:
            raise ValueError(
                f"target_sync_steps doit être > 0, got {self.target_sync_steps}"
            )
        if self.mutation_std < 0:
            raise ValueError(
                f"mutation_std doit être >= 0, got {self.mutation_std}"
            )
        if self.vision_radius < 1:
            raise ValueError(
                f"vision_radius doit être >= 1, got {self.vision_radius}"
            )


class LineageBrain:
    """Cerveau DQN partagé par tous les vivants d'une lignée."""

    def __init__(
        self,
        root_id: int,
        obs_dim: int,
        n_actions: int,
        cfg: BrainConfig,
        *,
        seed: int = 0,
    ) -> None:
        # Imports lazy pour éviter coût torch en collection-time
        import torch
        from mw_ia.neural.network import QNetwork
        from mw_ia.neural.replay_buffer import ReplayBuffer
        from mw_ia.neural.trainer import DQNTrainer

        self.root_id = root_id
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        wants_cuda = cfg.device == "cuda" and torch.cuda.is_available()
        self._torch = torch
        self.device = torch.device("cuda" if wants_cuda else "cpu")
        self.online = QNetwork(obs_dim, n_actions, cfg.hidden_dims).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, cfg.hidden_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.trainer = DQNTrainer(
            self.online, self.target, lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=False,
        )
        self.buffer = ReplayBuffer(cfg.buffer_capacity, obs_dim, seed=seed)
        self.global_step: int = 0
        self.target_syncs: int = 0
        self.last_loss: float | None = None

    @property
    def epsilon(self) -> float:
        if self.cfg.epsilon_decay_steps <= 0:
            return self.cfg.epsilon_end
        frac = min(1.0, self.global_step / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        """Epsilon-greedy si pas greedy, sinon argmax pure."""
        if (not greedy) and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            q = self.online(x)
            return int(q.argmax(dim=1).item())

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        """Push transition + train si conditions OK."""
        self.buffer.push(
            obs.astype(np.float32), action, reward,
            next_obs.astype(np.float32), done,
        )
        self.global_step += 1
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        if (
            len(self.buffer) >= self.cfg.min_replay_to_learn
            and self.global_step % self.cfg.train_every == 0
        ):
            batch = self.buffer.sample(self.cfg.batch_size)
            self.last_loss = self.trainer.step(batch)
            metrics["loss"] = self.last_loss
        if self.global_step % self.cfg.target_sync_steps == 0:
            self.trainer.sync_target()
            self.target_syncs += 1
        return metrics

    @classmethod
    def inherit_from(
        cls,
        parent: "LineageBrain",
        root_id: int,
        *,
        mutation_std: float | None = None,
        seed: int = 0,
    ) -> "LineageBrain":
        """Clone le cerveau parent + mutation gaussienne sur les poids.

        L'enfant démarre avec :
            - même architecture (obs_dim, n_actions, hidden_dims)
            - poids = parent + N(0, mutation_std)
            - buffer vide (pas hérité)
            - global_step = 0
            - target network = online (resync)
        """
        if mutation_std is None:
            mutation_std = parent.cfg.mutation_std
        child = cls(
            root_id=root_id,
            obs_dim=parent.obs_dim,
            n_actions=parent.n_actions,
            cfg=parent.cfg,
            seed=seed,
        )
        # Copy + mutation
        torch = child._torch
        with torch.no_grad():
            child.online.load_state_dict(parent.online.state_dict())
            if mutation_std > 0:
                for p in child.online.parameters():
                    noise = torch.randn_like(p) * mutation_std
                    p.add_(noise)
            # Resync target sur online (pas le target parent)
            child.target.load_state_dict(child.online.state_dict())
        return child

    def __repr__(self) -> str:
        return (
            f"LineageBrain(root_id={self.root_id}, obs_dim={self.obs_dim}, "
            f"n_actions={self.n_actions}, global_step={self.global_step}, "
            f"buffer={len(self.buffer)}, eps={self.epsilon:.3f})"
        )
