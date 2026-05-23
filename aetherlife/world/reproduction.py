"""ReproductionConfig + LineageEdge — composants V4 pour reproduction automatique.

Pattern V4 minimal : reproduction **automatique** quand l'agent atteint un seuil
d'énergie (pas une action choisie par l'agent). Cela évite de modifier l'action
space des agents RL existants tout en activant l'évolution darwinienne.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReproductionConfig:
    """Configuration de la reproduction automatique V4.

    Args:
        enabled: si False (default), aucune reproduction. Compat V2/V3.
        energy_threshold: parent doit avoir energy >= ce niveau pour se reproduire.
        energy_cost: énergie transférée au child (parent perd ce montant).
        cooldown_ticks: minimum de ticks entre 2 reproductions d'un même parent.
        max_population: cap population (cap dur, pas de nouvelles naissances au-delà).
    """

    enabled: bool = False
    energy_threshold: float = 80.0
    energy_cost: float = 40.0
    cooldown_ticks: int = 30
    max_population: int = 100

    def __post_init__(self) -> None:
        if self.energy_threshold <= 0:
            raise ValueError(
                f"energy_threshold doit être > 0 (got {self.energy_threshold})"
            )
        if self.energy_cost <= 0:
            raise ValueError(
                f"energy_cost doit être > 0 (got {self.energy_cost})"
            )
        if self.energy_cost >= self.energy_threshold:
            raise ValueError(
                f"energy_cost ({self.energy_cost}) doit être < "
                f"energy_threshold ({self.energy_threshold})"
            )
        if self.cooldown_ticks < 0:
            raise ValueError(
                f"cooldown_ticks doit être >= 0 (got {self.cooldown_ticks})"
            )
        if self.max_population <= 0:
            raise ValueError(
                f"max_population doit être > 0 (got {self.max_population})"
            )


@dataclass(frozen=True)
class LineageEdge:
    """Arête parent → enfant dans le lineage."""

    parent_id: int
    child_id: int
    birth_tick: int
    parent_generation: int
    child_generation: int
