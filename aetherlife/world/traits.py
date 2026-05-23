"""TraitsConfig + AgentTraits — V7 mini-évolution darwinienne sans RL.

Chaque agent porte des **biais comportementaux** (build / plant / cache / explore)
dans [0, 1] qui pondèrent ses heuristiques. À la reproduction, l'enfant hérite
des traits du parent avec une mutation gaussienne bornée.

Pattern V7 : pas d'apprentissage individuel, mais sélection naturelle des
lignées. Les bons traits propagent, les mauvais s'éteignent → "évolution"
émergente visible sur 1000+ ticks.

Vocabulaire :
    - build_bias    : tendance à construire un nid (multiplie la priorité)
    - plant_bias    : tendance à planter une graine
    - cache_bias    : tendance à déposer/retirer du cache (vs manger direct)
    - explore_bias  : tendance à errer hors-zone (vs zone connue)

Tous les biais sont dans [0, 1]. 0.5 = comportement neutre / baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TraitsConfig:
    """Configuration de l'héritage de traits + mutation.

    Args:
        enabled: si False (default), pas de traits (compat V6 et avant).
        mutation_std: écart-type gaussien appliqué à chaque trait à l'héritage.
        mutation_clamp: borne min/max après mutation (traits ∈ [0, 1] toujours).
        initial_mean: moyenne pour les traits des agents initiaux.
        initial_std: écart-type pour les traits des agents initiaux.
    """

    enabled: bool = False
    mutation_std: float = 0.08
    mutation_clamp: tuple[float, float] = (0.0, 1.0)
    initial_mean: float = 0.5
    initial_std: float = 0.15

    def __post_init__(self) -> None:
        if self.mutation_std < 0:
            raise ValueError(f"mutation_std doit être >= 0 (got {self.mutation_std})")
        lo, hi = self.mutation_clamp
        if lo >= hi:
            raise ValueError(
                f"mutation_clamp = (lo={lo}, hi={hi}) doit avoir lo < hi"
            )
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"mutation_clamp doit être ⊆ [0, 1] (got {self.mutation_clamp})"
            )
        if not (lo <= self.initial_mean <= hi):
            raise ValueError(
                f"initial_mean ({self.initial_mean}) doit être dans clamp "
                f"({self.mutation_clamp})"
            )
        if self.initial_std < 0:
            raise ValueError(f"initial_std doit être >= 0 (got {self.initial_std})")


@dataclass(frozen=True)
class AgentTraits:
    """Biais comportementaux d'un agent. Tous dans [0, 1]."""

    build_bias: float = 0.5
    plant_bias: float = 0.5
    cache_bias: float = 0.5
    explore_bias: float = 0.5

    def __post_init__(self) -> None:
        for name in ("build_bias", "plant_bias", "cache_bias", "explore_bias"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} doit être dans [0, 1] (got {v})")

    def as_array(self) -> np.ndarray:
        """Vecteur (4,) pour télémétrie / debugging."""
        return np.array(
            [self.build_bias, self.plant_bias, self.cache_bias, self.explore_bias],
            dtype=np.float32,
        )

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        cfg: TraitsConfig,
    ) -> "AgentTraits":
        """Tire 4 traits gaussiens N(initial_mean, initial_std), clampés."""
        lo, hi = cfg.mutation_clamp
        vals = rng.normal(cfg.initial_mean, cfg.initial_std, size=4)
        vals = np.clip(vals, lo, hi)
        return cls(
            build_bias=float(vals[0]),
            plant_bias=float(vals[1]),
            cache_bias=float(vals[2]),
            explore_bias=float(vals[3]),
        )

    def mutate(
        self,
        rng: np.random.Generator,
        cfg: TraitsConfig,
    ) -> "AgentTraits":
        """Renvoie un nouveau AgentTraits muté gaussiennement."""
        lo, hi = cfg.mutation_clamp
        noise = rng.normal(0.0, cfg.mutation_std, size=4)
        vals = np.array(
            [self.build_bias, self.plant_bias, self.cache_bias, self.explore_bias]
        ) + noise
        vals = np.clip(vals, lo, hi)
        return AgentTraits(
            build_bias=float(vals[0]),
            plant_bias=float(vals[1]),
            cache_bias=float(vals[2]),
            explore_bias=float(vals[3]),
        )


@dataclass
class TraitDistribution:
    """Snapshot agrégé des traits d'une population vivante (pour telemetry)."""

    n: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    std: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    @classmethod
    def from_traits(cls, traits: list[AgentTraits]) -> "TraitDistribution":
        if not traits:
            return cls()
        arr = np.array([t.as_array() for t in traits], dtype=np.float32)
        return cls(
            n=len(traits),
            mean=arr.mean(axis=0),
            std=arr.std(axis=0),
        )
