"""Tests pour les invariants Python — strictement les mêmes examples que les .aether."""
from __future__ import annotations

import pytest

from aetherlife.guardrails.exceptions import InvariantViolationError
from aetherlife.guardrails.invariants import (
    clamp_pos,
    energy_no_food,
    energy_with_food,
    is_terminated,
    step_reward,
)


@pytest.mark.parametrize(
    "energy,metabolism,expected",
    [
        (10, 1, 9),
        (5, 3, 2),
        (1, 1, 0),
        (0, 1, 0),
        (2, 5, 0),
    ],
)
def test_i1_energy_no_food(energy: float, metabolism: float, expected: float) -> None:
    """I1 — examples strictement identiques à i1_energy_no_food.aether."""
    assert energy_no_food(energy, metabolism) == expected


def test_i1_invariants() -> None:
    """I1 — 0 ≤ result ≤ energy pour tout (energy ≥ 0, metabolism ≥ 0)."""
    for energy in [0, 1, 5, 10, 100]:
        for metabolism in [0, 1, 5, 50]:
            r = energy_no_food(energy, metabolism)
            assert r >= 0
            assert r <= energy


@pytest.mark.parametrize(
    "energy,metabolism,food_value,max_energy,expected",
    [
        (10, 1, 5, 20, 14),
        (18, 1, 5, 20, 20),
        (19, 1, 5, 20, 20),
        (0, 1, 5, 20, 4),
        (1, 5, 5, 20, 1),
    ],
)
def test_i2_energy_with_food(
    energy: float, metabolism: float, food_value: float, max_energy: float, expected: float
) -> None:
    """I2 — examples strictement identiques à i2_energy_with_food.aether."""
    assert energy_with_food(energy, metabolism, food_value, max_energy) == expected


def test_i2_invariants() -> None:
    """I2 — 0 ≤ result ≤ max_energy."""
    for energy in [0, 10, 50, 100]:
        for metab in [0, 1, 10]:
            for fv in [0, 5, 20]:
                for me in [10, 50, 100]:
                    r = energy_with_food(energy, metab, fv, me)
                    assert 0 <= r <= me


@pytest.mark.parametrize(
    "energy,expected",
    [(0, True), (-1, True), (1, False), (100, False)],
)
def test_i3_is_terminated(energy: float, expected: bool) -> None:
    """I3 — examples strictement identiques à i3_terminated.aether."""
    assert is_terminated(energy) is expected


@pytest.mark.parametrize(
    "metabolism,food_value,ate,expected",
    [
        (1, 5, True, 4),
        (1, 5, False, -1),
        (2, 10, True, 8),
        (2, 10, False, -2),
    ],
)
def test_i4_step_reward(
    metabolism: float, food_value: float, ate: bool, expected: float
) -> None:
    """I4 — examples strictement identiques à i4_step_reward.aether."""
    assert step_reward(metabolism, food_value, ate) == expected


@pytest.mark.parametrize(
    "pos,delta,dim,expected",
    [
        (5, 1, 10, 6),
        (9, 1, 10, 9),
        (0, -1, 10, 0),
        (3, 0, 10, 3),
        (9, 5, 10, 9),
    ],
)
def test_i5_clamp_pos(pos: int, delta: int, dim: int, expected: int) -> None:
    """I5 — examples strictement identiques à i5_clamp_pos.aether."""
    assert clamp_pos(pos, delta, dim) == expected


def test_i5_invariants() -> None:
    """I5 — 0 ≤ result < dim."""
    for pos in range(10):
        for delta in [-5, -1, 0, 1, 5]:
            r = clamp_pos(pos, delta, 10)
            assert 0 <= r < 10


def test_invariant_violation_error_format() -> None:
    err = InvariantViolationError("I1", "energy went negative", {"energy": -5})
    assert "I1" in str(err)
    assert "energy went negative" in str(err)
    assert "energy" in str(err)
