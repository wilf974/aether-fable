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


def pop_after_deaths(n_alive_before: int, n_died: int) -> int:
    """I6 — sans naissance, population vivante après step ≤ population avant.

    Mirror de `aether/invariants/i6_pop_after_deaths.aether`.
    Invariants : 0 ≤ result ≤ n_alive_before.
    """
    return max(0, n_alive_before - n_died)


def energy_gained(n_food_eaten: int, food_value: float) -> float:
    """I7 — conservation : énergie gagnée = food consommée × food_value.

    Mirror de `aether/invariants/i7_energy_gained.aether`.
    Invariants : result ≥ 0 et result = n_food_eaten × food_value.
    """
    return n_food_eaten * food_value


def total_ids_emitted(n_total_ids_alive: int, n_dead_so_far: int) -> int:
    """I8 — total des agent_ids émis pendant l'épisode.

    Pas de réutilisation : un agent mort ne libère pas son id.
    Mirror de `aether/invariants/i8_total_ids_emitted.aether`.
    """
    return n_total_ids_alive + n_dead_so_far


def season_phase(step_count: int, season_period: int) -> float:
    """I9 — season_phase ∈ [0, 1) après chaque tick.

    Mirror de `aether/invariants/i9_season_phase.aether`.
    Invariants : 0 ≤ result < 1.
    """
    return (step_count % season_period) / season_period


def clamp_temp(temp: float, temp_min: float, temp_max: float) -> float:
    """I10 — température clampée dans [temp_min, temp_max].

    Mirror de `aether/invariants/i10_clamp_temp.aether`.
    Invariants : temp_min ≤ result ≤ temp_max.
    """
    return max(temp_min, min(temp_max, temp))


def seasonal_lambda(base_lambda: float, season_factor: float) -> float:
    """I11 — lambda saisonnier toujours ≥ 0 (modulation positive).

    Mirror de `aether/invariants/i11_seasonal_lambda.aether`.
    Invariants : result ≥ 0.
    """
    return max(0.0, base_lambda * season_factor)


def child_generation(parent_gen: int) -> int:
    """I12 — generation enfant = generation parent + 1.

    Mirror de `aether/invariants/i12_child_generation.aether`.
    Invariants : result = parent_gen + 1 (strict monotone).
    """
    return parent_gen + 1


def child_birth_tick(parent_birth_tick: int, current_tick: int) -> int:
    """I13 — birth_tick enfant = current_tick ≥ birth_tick parent.

    Mirror de `aether/invariants/i13_child_birth_tick.aether`.
    Garantit que le lineage est non-acyclique en temps.
    """
    return current_tick


def pop_after_births(pop_before: int, n_births: int, max_pop: int) -> int:
    """I14 — population bornée par max_pop après naissances.

    Mirror de `aether/invariants/i14_pop_after_births.aether`.
    Invariants : pop_before ≤ result ≤ max_pop.
    """
    return min(max_pop, pop_before + n_births)


def rest_energy_gain(
    energy_before: float, rest_bonus: float, max_energy: float
) -> float:
    """I15 — gain énergie sur nid : clamp à max_energy, jamais < energy_before.

    Mirror de `aether/invariants/i15_rest_energy_gain.aether`.
    Invariants : energy_before ≤ result ≤ max_energy (avec rest_bonus ≥ 0).
    """
    return min(max_energy, energy_before + rest_bonus)


def nests_after_build(current_nests: int, built_this_tick: int) -> int:
    """I16 — nb de nids par agent ≤ 1.

    Mirror de `aether/invariants/i16_nests_after_build.aether`.
    Invariants : 0 ≤ result ≤ 1.
    """
    return min(1, current_nests + built_this_tick)


def energy_after_build(energy_before: float, build_cost: float) -> float:
    """I17 — énergie après construction = energy_before - build_cost, ≥ 0.

    Mirror de `aether/invariants/i17_energy_after_build.aether`.
    Précondition (à enforcer côté caller) : energy_before ≥ build_cost.
    """
    return max(0.0, energy_before - build_cost)
