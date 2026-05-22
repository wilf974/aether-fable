"""Configurations frozen dataclasses pour AetherLife V1."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FoodGridConfig:
    """Configuration de l'environnement V1 — solo forager.

    Tous les champs ont des defaults validés en __post_init__.
    """

    rows: int = 16
    cols: int = 16
    max_energy: float = 100.0
    start_energy: float = 50.0
    metabolism: float = 1.0
    food_value: float = 20.0
    death_penalty: float = 50.0
    initial_food_density: float = 0.05
    food_respawn_lambda: float = 0.5
    max_steps: int = 1000
    start_position: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"rows et cols doivent être > 0 (got {self.rows}, {self.cols})")
        if self.max_energy <= 0:
            raise ValueError(f"max_energy doit être > 0 (got {self.max_energy})")
        if not (0 < self.start_energy <= self.max_energy):
            raise ValueError(
                f"start_energy doit être dans (0, max_energy={self.max_energy}] "
                f"(got {self.start_energy})"
            )
        if self.metabolism <= 0:
            raise ValueError(f"metabolism doit être > 0 (got {self.metabolism})")
        if self.food_value <= 0:
            raise ValueError(f"food_value doit être > 0 (got {self.food_value})")
        if self.death_penalty < 0:
            raise ValueError(f"death_penalty doit être >= 0 (got {self.death_penalty})")
        if not (0.0 <= self.initial_food_density <= 1.0):
            raise ValueError(
                f"initial_food_density doit être dans [0, 1] (got {self.initial_food_density})"
            )
        if self.food_respawn_lambda < 0:
            raise ValueError(
                f"food_respawn_lambda doit être >= 0 (got {self.food_respawn_lambda})"
            )
        if self.max_steps <= 0:
            raise ValueError(f"max_steps doit être > 0 (got {self.max_steps})")
        r, c = self.start_position
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(
                f"start_position {self.start_position} hors grille "
                f"({self.rows}x{self.cols})"
            )

    @property
    def obs_dim(self) -> int:
        """Dimension de l'observation = position_one_hot + food_flatten + energy_normalized."""
        return 2 * self.rows * self.cols + 1
