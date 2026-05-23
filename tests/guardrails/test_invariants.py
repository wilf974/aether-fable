"""Tests pour les invariants Python — strictement les mêmes examples que les .aether."""
from __future__ import annotations

import pytest

from aetherlife.guardrails.exceptions import InvariantViolationError
from aetherlife.guardrails.invariants import (
    child_birth_tick,
    child_generation,
    clamp_pos,
    clamp_temp,
    energy_after_build,
    energy_gained,
    energy_no_food,
    energy_with_food,
    is_terminated,
    nests_after_build,
    pop_after_births,
    pop_after_deaths,
    rest_energy_gain,
    season_phase,
    seasonal_lambda,
    step_reward,
    total_ids_emitted,
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


@pytest.mark.parametrize(
    "n_alive_before,n_died,expected",
    [(10, 0, 10), (10, 1, 9), (10, 3, 7), (5, 5, 0), (0, 0, 0)],
)
def test_i6_pop_after_deaths(n_alive_before: int, n_died: int, expected: int) -> None:
    """I6 — examples strictement identiques à i6_pop_after_deaths.aether."""
    assert pop_after_deaths(n_alive_before, n_died) == expected


def test_i6_invariants() -> None:
    """I6 — 0 ≤ result ≤ n_alive_before."""
    for n in range(0, 20):
        for d in range(0, n + 5):
            r = pop_after_deaths(n, d)
            assert 0 <= r <= n


@pytest.mark.parametrize(
    "n_food_eaten,food_value,expected",
    [(0, 5, 0), (1, 5, 5), (3, 5, 15), (10, 20, 200)],
)
def test_i7_energy_gained(n_food_eaten: int, food_value: float, expected: float) -> None:
    """I7 — examples strictement identiques à i7_energy_gained.aether."""
    assert energy_gained(n_food_eaten, food_value) == expected


@pytest.mark.parametrize(
    "n_alive,n_dead,expected",
    [(5, 3, 8), (0, 5, 5), (10, 0, 10)],
)
def test_i8_total_ids_emitted(n_alive: int, n_dead: int, expected: int) -> None:
    """I8 — examples strictement identiques à i8_total_ids_emitted.aether."""
    assert total_ids_emitted(n_alive, n_dead) == expected


@pytest.mark.parametrize(
    "step_count,season_period,expected",
    [
        (0, 200, 0.0),
        (50, 200, 0.25),
        (100, 200, 0.5),
        (199, 200, 0.995),
        (200, 200, 0.0),
        (400, 200, 0.0),
        (250, 200, 0.25),
    ],
)
def test_i9_season_phase(step_count: int, season_period: int, expected: float) -> None:
    """I9 — examples strictement identiques à i9_season_phase.aether."""
    assert season_phase(step_count, season_period) == expected


def test_i9_invariants() -> None:
    """I9 — 0 ≤ result < 1."""
    for step in range(0, 1000, 13):
        for period in [10, 50, 200, 365]:
            r = season_phase(step, period)
            assert 0 <= r < 1


@pytest.mark.parametrize(
    "temp,tmin,tmax,expected",
    [
        (-50, -10, 30, -10),
        (50, -10, 30, 30),
        (15, -10, 30, 15),
        (-10, -10, 30, -10),
        (30, -10, 30, 30),
    ],
)
def test_i10_clamp_temp(temp: float, tmin: float, tmax: float, expected: float) -> None:
    """I10 — examples strictement identiques à i10_clamp_temp.aether."""
    assert clamp_temp(temp, tmin, tmax) == expected


@pytest.mark.parametrize(
    "base,factor,expected",
    [(1.0, 0.5, 0.5), (1.0, 0.0, 0.0), (2.0, 1.5, 3.0), (0, 1.0, 0), (1.0, -0.5, 0.0)],
)
def test_i11_seasonal_lambda(base: float, factor: float, expected: float) -> None:
    """I11 — examples + cas négatif (clampé à 0)."""
    assert seasonal_lambda(base, factor) == expected


@pytest.mark.parametrize("parent_gen,expected", [(0, 1), (1, 2), (5, 6), (10, 11)])
def test_i12_child_generation(parent_gen: int, expected: int) -> None:
    """I12 — examples strictement identiques à i12_child_generation.aether."""
    assert child_generation(parent_gen) == expected


def test_i12_invariants() -> None:
    """I12 — result = parent_gen + 1 (monotone strict)."""
    for g in range(0, 100):
        r = child_generation(g)
        assert r == g + 1
        assert r > g


@pytest.mark.parametrize(
    "parent_birth,current,expected",
    [(0, 10, 10), (5, 100, 100), (50, 60, 60)],
)
def test_i13_child_birth_tick(parent_birth: int, current: int, expected: int) -> None:
    """I13 — examples strictement identiques à i13_child_birth_tick.aether."""
    assert child_birth_tick(parent_birth, current) == expected


@pytest.mark.parametrize(
    "pop_before,n_births,max_pop,expected",
    [(10, 3, 100, 13), (50, 60, 100, 100), (0, 5, 10, 5), (95, 10, 100, 100)],
)
def test_i14_pop_after_births(
    pop_before: int, n_births: int, max_pop: int, expected: int
) -> None:
    """I14 — examples strictement identiques à i14_pop_after_births.aether."""
    assert pop_after_births(pop_before, n_births, max_pop) == expected


def test_i14_invariants() -> None:
    """I14 — pop_before ≤ result ≤ max_pop."""
    for pb in [0, 10, 50, 90]:
        for nb in [0, 5, 50]:
            for mp in [100, 200]:
                r = pop_after_births(pb, nb, mp)
                assert pb <= r <= mp


@pytest.mark.parametrize(
    "energy,bonus,max_e,expected",
    [(50, 5, 100, 55), (95, 10, 100, 100), (99, 5, 100, 100), (0, 5, 100, 5)],
)
def test_i15_rest_energy_gain(
    energy: float, bonus: float, max_e: float, expected: float
) -> None:
    """I15 — examples strictement identiques à i15_rest_energy_gain.aether."""
    assert rest_energy_gain(energy, bonus, max_e) == expected


def test_i15_invariants() -> None:
    """I15 — energy_before ≤ result ≤ max_energy."""
    for e in [0, 10, 50, 90, 99]:
        for b in [0, 5, 20]:
            for me in [100, 200]:
                r = rest_energy_gain(e, b, me)
                assert e <= r <= me


@pytest.mark.parametrize(
    "current,built,expected",
    [(0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 0)],
)
def test_i16_nests_after_build(current: int, built: int, expected: int) -> None:
    """I16 — examples strictement identiques à i16_nests_after_build.aether."""
    assert nests_after_build(current, built) == expected


@pytest.mark.parametrize(
    "energy,cost,expected",
    [(100, 20, 80), (50, 30, 20), (30, 30, 0)],
)
def test_i17_energy_after_build(energy: float, cost: float, expected: float) -> None:
    """I17 — examples strictement identiques à i17_energy_after_build.aether."""
    assert energy_after_build(energy, cost) == expected


def test_invariant_violation_error_format() -> None:
    err = InvariantViolationError("I1", "energy went negative", {"energy": -5})
    assert "I1" in str(err)
    assert "energy went negative" in str(err)
    assert "energy" in str(err)
