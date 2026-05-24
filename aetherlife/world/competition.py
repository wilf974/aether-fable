"""V8-B1.5 — Compétition locale : metabolism modulé par densité voisine.

Hypothèse écologique : si un agent a beaucoup de voisins proches, il
brûle plus d'énergie (concurrence pour ressources locales). Cela force
les lignées à se disperser dans des zones moins peuplées au lieu de
toutes se regrouper dans le biome optimal.

Effet attendu : disperse les lignées, ouvre des niches spatiales, et
empêche le monopole 100% observé en V8-B1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionConfig:
    """Configuration de la compétition locale V8-B1.5."""

    enabled: bool = False
    radius: int = 3                       # voisins dans Manhattan ≤ radius
    metabolism_per_neighbor: float = 0.04  # +4% metabolism par voisin proche
    max_factor: float = 2.5                # cap à 2.5× metabolism

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError(f"radius doit être >= 0 (got {self.radius})")
        if self.metabolism_per_neighbor < 0:
            raise ValueError(
                f"metabolism_per_neighbor doit être >= 0 "
                f"(got {self.metabolism_per_neighbor})"
            )
        if self.max_factor < 1.0:
            raise ValueError(
                f"max_factor doit être >= 1.0 (got {self.max_factor})"
            )


def crowd_metabolism_factor(
    n_neighbors: int, cfg: CompetitionConfig,
) -> float:
    """Retourne le multiplicateur de metabolism dû à la densité voisine."""
    if not cfg.enabled or n_neighbors <= 0:
        return 1.0
    factor = 1.0 + n_neighbors * cfg.metabolism_per_neighbor
    return min(factor, cfg.max_factor)
