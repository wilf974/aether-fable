"""Mirror Python des invariants Aether V1 (I1-I5).

Chaque fonction est strictement isomorphique à sa contrepartie .aether dans
aether/invariants/. Toute divergence est un bug.
"""
from __future__ import annotations


def energy_no_food(energy: float, metabolism: float) -> float:
    """I1 — énergie après step sans manger : energy - metabolism, clampée à 0.

    Mirror de `aether/invariants/i1_energy_no_food.aether`.
    Invariants : 0 ≤ result ≤ energy.
    """
    return max(0.0, energy - metabolism)


def energy_with_food(
    energy: float, metabolism: float, food_value: float, max_energy: float
) -> float:
    """I2 — énergie après step avec eat : clamp(energy - metabolism + food_value, 0, max_energy).

    Mirror de `aether/invariants/i2_energy_with_food.aether`.
    Invariants : 0 ≤ result ≤ max_energy.
    """
    return max(0.0, min(max_energy, energy - metabolism + food_value))


def is_terminated(energy: float) -> bool:
    """I3 — terminated ssi energy ≤ 0.

    Mirror de `aether/invariants/i3_terminated.aether`.
    """
    return energy <= 0


def step_reward(metabolism: float, food_value: float, ate: bool) -> float:
    """I4 — reward d'un step : -metabolism + (food_value si ate).

    Mirror de `aether/invariants/i4_step_reward.aether`.
    """
    if ate:
        return food_value - metabolism
    return -metabolism


def clamp_pos(pos: int, delta: int, dim: int) -> int:
    """I5 — position clampée dans [0, dim-1] après step (1D).

    Mirror de `aether/invariants/i5_clamp_pos.aether`.
    Invariants : 0 ≤ result < dim.
    """
    return max(0, min(dim - 1, pos + delta))
