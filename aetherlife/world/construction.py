"""BuildConfig + NestRecord — composants V5 pour construction automatique.

Pattern V5 minimal : construction **automatique** quand un agent a une énergie
suffisante et a passé X ticks sans construire. Un agent ne peut posséder
**qu'un seul nid**. Quand il revient sur son nid il gagne `rest_bonus`
d'énergie (clampée à `max_energy`). Le nid disparaît à la mort de l'agent.

Cohérent avec la mécanique V4 (reproduction auto) — pas de changement d'action
space, compat des wrappers RL existants préservée.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildConfig:
    """Configuration de la construction automatique V5.

    Args:
        enabled: si False (default), aucune construction. Compat V2/V3/V4.
        energy_threshold: agent doit avoir energy ≥ ce niveau pour construire.
        build_cost: énergie consommée pour la construction.
        rest_bonus: énergie regagnée à chaque tick passé sur son propre nid.
        cooldown_ticks: minimum de ticks entre 2 tentatives de construction.
    """

    enabled: bool = False
    energy_threshold: float = 90.0
    build_cost: float = 25.0
    rest_bonus: float = 3.0
    cooldown_ticks: int = 50

    def __post_init__(self) -> None:
        if self.energy_threshold <= 0:
            raise ValueError(
                f"energy_threshold doit être > 0 (got {self.energy_threshold})"
            )
        if self.build_cost <= 0:
            raise ValueError(f"build_cost doit être > 0 (got {self.build_cost})")
        if self.build_cost >= self.energy_threshold:
            raise ValueError(
                f"build_cost ({self.build_cost}) doit être < "
                f"energy_threshold ({self.energy_threshold})"
            )
        if self.rest_bonus < 0:
            raise ValueError(f"rest_bonus doit être >= 0 (got {self.rest_bonus})")
        if self.cooldown_ticks < 0:
            raise ValueError(
                f"cooldown_ticks doit être >= 0 (got {self.cooldown_ticks})"
            )


@dataclass(frozen=True)
class NestRecord:
    """Nid construit par un agent — propre à un seul owner."""

    owner_id: int
    pos: tuple[int, int]
    built_tick: int
