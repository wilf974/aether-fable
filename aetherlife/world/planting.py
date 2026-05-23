"""V6 — système de plantation : agents cultivent food sur cellules vides.

Mécanique :
- Un agent avec energy >= `energy_threshold` peut planter sur sa cellule
  actuelle si :
    - cellule libre de food et de nid
    - pas déjà une plante en croissance ici
    - cooldown de plantation respecté
- Coût : `energy_cost` (déduit de l'agent)
- La graine pousse pendant `growth_ticks` puis devient une food cell normale
  (consommable par n'importe quel agent)
- Visualisation : taille croissante d'un petit cercle vert pâle

Tension introduite :
    chasse passive (food spontanée)
    vs
    investissement actif (planter → récolter)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlantingConfig:
    """Configuration de la plantation V6.

    Args:
        enabled: si False (default), aucune plantation. Compat V5/V5.2/V5.3.
        energy_threshold: minimum d'énergie pour pouvoir planter.
        energy_cost: énergie consommée par plantation.
        growth_ticks: nombre de ticks pour qu'une plante devienne food.
        cooldown_ticks: ticks min entre 2 plantations du même agent.
    """

    enabled: bool = False
    energy_threshold: float = 100.0
    energy_cost: float = 12.0
    growth_ticks: int = 40
    cooldown_ticks: int = 25

    def __post_init__(self) -> None:
        if self.energy_threshold <= 0:
            raise ValueError(
                f"energy_threshold doit être > 0 (got {self.energy_threshold})"
            )
        if self.energy_cost <= 0:
            raise ValueError(f"energy_cost doit être > 0 (got {self.energy_cost})")
        if self.energy_cost >= self.energy_threshold:
            raise ValueError(
                f"energy_cost ({self.energy_cost}) doit être < "
                f"energy_threshold ({self.energy_threshold})"
            )
        if self.growth_ticks <= 0:
            raise ValueError(f"growth_ticks doit être > 0 (got {self.growth_ticks})")
        if self.cooldown_ticks < 0:
            raise ValueError(f"cooldown_ticks doit être >= 0 (got {self.cooldown_ticks})")


@dataclass
class PlantRecord:
    """Plante en croissance sur une cellule (V6)."""

    planter_id: int
    pos: tuple[int, int]
    planted_tick: int
    matures_at_tick: int

    def progress(self, current_tick: int) -> float:
        """Fraction de croissance dans [0, 1]."""
        total = self.matures_at_tick - self.planted_tick
        if total <= 0:
            return 1.0
        return min(1.0, max(0.0, (current_tick - self.planted_tick) / total))

    def is_mature(self, current_tick: int) -> bool:
        return current_tick >= self.matures_at_tick
