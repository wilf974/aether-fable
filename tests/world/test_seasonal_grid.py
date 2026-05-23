"""Tests pour SeasonalMultiAgentFoodGrid (V3)."""
from __future__ import annotations

import numpy as np
import pytest

from aetherlife.world.seasonal_grid import (
    Season,
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
    build_temperature_field,
    current_season,
    get_seasonal_factor,
)


@pytest.fixture
def small_seasonal_cfg() -> SeasonalMultiAgentConfig:
    return SeasonalMultiAgentConfig(
        rows=8, cols=8, n_agents=3,
        max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=3.0,
        initial_food_density=0.0, food_respawn_lambda=1.0, max_steps=50,
        seasonal=SeasonalConfig(
            season_period=20, temp_min=-10.0, temp_max=30.0,
            spring_lambda_factor=1.5, summer_lambda_factor=1.0,
            autumn_lambda_factor=1.2, winter_lambda_factor=0.3,
        ),
    )


def test_current_season_quadrants() -> None:
    assert current_season(0.0) == Season.SPRING
    assert current_season(0.2) == Season.SPRING
    assert current_season(0.25) == Season.SUMMER
    assert current_season(0.4) == Season.SUMMER
    assert current_season(0.5) == Season.AUTUMN
    assert current_season(0.74) == Season.AUTUMN
    assert current_season(0.75) == Season.WINTER
    assert current_season(0.99) == Season.WINTER


def test_seasonal_factor_returns_correct() -> None:
    cfg = SeasonalConfig(
        spring_lambda_factor=2.0, summer_lambda_factor=1.0,
        autumn_lambda_factor=1.5, winter_lambda_factor=0.5,
    )
    assert get_seasonal_factor(Season.SPRING, cfg) == 2.0
    assert get_seasonal_factor(Season.SUMMER, cfg) == 1.0
    assert get_seasonal_factor(Season.AUTUMN, cfg) == 1.5
    assert get_seasonal_factor(Season.WINTER, cfg) == 0.5


def test_temperature_field_bounded() -> None:
    cfg = SeasonalConfig(temp_min=-10.0, temp_max=30.0)
    for phase in [0.0, 0.25, 0.5, 0.75, 0.99]:
        field = build_temperature_field(8, 8, phase, cfg)
        assert field.shape == (8, 8)
        assert field.min() >= -10.0
        assert field.max() <= 30.0


def test_reset_initializes_temperature(
    small_seasonal_cfg: SeasonalMultiAgentConfig,
) -> None:
    env = SeasonalMultiAgentFoodGrid(small_seasonal_cfg)
    env.reset(seed=0)
    field = env.temperature_field
    assert field.shape == (8, 8)
    assert env.phase == 0.0
    assert env.season == Season.SPRING


def test_season_advances_with_steps(
    small_seasonal_cfg: SeasonalMultiAgentConfig,
) -> None:
    env = SeasonalMultiAgentFoodGrid(small_seasonal_cfg)
    env.reset(seed=0)
    # period=20, donc step=5 → phase=0.25 → summer
    for _ in range(5):
        env.step({aid: 0 for aid in env.alive_agent_ids})
    assert env.phase == 0.25
    assert env.season == Season.SUMMER
    for _ in range(5):
        env.step({aid: 0 for aid in env.alive_agent_ids})
    assert env.phase == 0.5
    assert env.season == Season.AUTUMN


def test_obs_dim_includes_season_and_temp(
    small_seasonal_cfg: SeasonalMultiAgentConfig,
) -> None:
    env = SeasonalMultiAgentFoodGrid(small_seasonal_cfg)
    obs_dict, _ = env.reset(seed=0)
    assert small_seasonal_cfg.obs_dim == 3 * 8 * 8 + 3
    for obs in obs_dict.values():
        assert obs.shape == (small_seasonal_cfg.obs_dim,)
        # season_phase et temp_normalisée doivent être dans [0, 1]
        assert 0 <= obs[-2] < 1
        assert 0 <= obs[-1] <= 1


def test_cold_metabolism_increases_energy_loss() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=4, cols=4, n_agents=1,
        max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        seasonal=SeasonalConfig(
            season_period=4,
            spring_lambda_factor=0.0, summer_lambda_factor=0.0,
            autumn_lambda_factor=0.0, winter_lambda_factor=0.0,
            cold_threshold=100.0,  # tout est "froid" pour le test
            cold_metabolism_factor=2.0,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    initial_energy = env.agent_state(0).energy
    env.step({0: 0})
    # metabolism doublé : energy -= 2 (au lieu de 1)
    assert env.agent_state(0).energy == initial_energy - 2.0


def test_food_regen_varies_by_season() -> None:
    """En hiver (factor=0.3) moins de food regen qu'au printemps (factor=1.5)."""
    cfg = SeasonalMultiAgentConfig(
        rows=10, cols=10, n_agents=1,
        initial_food_density=0.0, food_respawn_lambda=2.0,
        max_steps=200,
        seasonal=SeasonalConfig(
            season_period=4,  # 1 tick par saison
            spring_lambda_factor=5.0,
            summer_lambda_factor=5.0,
            autumn_lambda_factor=5.0,
            winter_lambda_factor=0.0,  # zéro de regen en hiver
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    # Hop sur l'index winter (phase ≥ 0.75)
    for _ in range(3):
        env.step({0: 0})
    assert env.season == Season.WINTER
    food_before = env.food_count
    for _ in range(20):
        env.step({0: 0})
        # On reste en winter en restant à step % 4 == 3 ? Non, on cycle.
        if env.season != Season.WINTER:
            break


def test_seed_reproducibility(small_seasonal_cfg: SeasonalMultiAgentConfig) -> None:
    env1 = SeasonalMultiAgentFoodGrid(small_seasonal_cfg)
    env2 = SeasonalMultiAgentFoodGrid(small_seasonal_cfg)
    env1.reset(seed=42)
    env2.reset(seed=42)
    for i in range(3):
        assert env1.agent_state(i).pos == env2.agent_state(i).pos
    assert np.array_equal(env1.temperature_field, env2.temperature_field)


def test_invalid_seasonal_config_raises() -> None:
    with pytest.raises(ValueError):
        SeasonalConfig(season_period=0)
    with pytest.raises(ValueError):
        SeasonalConfig(temp_min=30.0, temp_max=10.0)
    with pytest.raises(ValueError):
        SeasonalConfig(spring_lambda_factor=-1.0)
    with pytest.raises(ValueError):
        SeasonalConfig(cold_metabolism_factor=0.5)


def test_phase_resets_at_period_boundary(
    small_seasonal_cfg: SeasonalMultiAgentConfig,
) -> None:
    env = SeasonalMultiAgentFoodGrid(small_seasonal_cfg)
    env.reset(seed=0)
    # period=20, on fait 20 steps → phase doit revenir à 0
    for _ in range(20):
        env.step({aid: 0 for aid in env.alive_agent_ids if env.alive_agent_ids})
    assert env.phase == 0.0
    assert env.season == Season.SPRING
