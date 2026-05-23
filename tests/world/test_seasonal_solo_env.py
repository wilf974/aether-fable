"""Tests pour SoloSeasonalEnv (V3.5)."""
from __future__ import annotations

import numpy as np
import pytest

from aetherlife.world.food_grid import Action
from aetherlife.world.seasonal_grid import SeasonalConfig, SeasonalMultiAgentConfig
from aetherlife.world.seasonal_solo_env import SoloSeasonalEnv


def test_solo_seasonal_reset_returns_obs_tuple() -> None:
    env = SoloSeasonalEnv(SeasonalMultiAgentConfig(
        rows=5, cols=5, n_agents=1, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=20,
        seasonal=SeasonalConfig(season_period=20),
    ))
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.cfg.obs_dim,)
    assert obs.dtype == np.float32
    assert "step" in info


def test_solo_seasonal_step_5tuple() -> None:
    env = SoloSeasonalEnv(SeasonalMultiAgentConfig(
        rows=5, cols=5, n_agents=1, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=10,
        seasonal=SeasonalConfig(season_period=10),
    ))
    env.reset(seed=0)
    result = env.step(int(Action.NORTH))
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(reward, float)
    assert "season" in info


def test_solo_seasonal_forces_n_agents_to_1() -> None:
    """Si on passe n_agents > 1, c'est forcé à 1."""
    env = SoloSeasonalEnv(SeasonalMultiAgentConfig(
        rows=5, cols=5, n_agents=5, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=10,
    ))
    assert env.cfg.n_agents == 1


def test_solo_seasonal_season_advances() -> None:
    env = SoloSeasonalEnv(SeasonalMultiAgentConfig(
        rows=4, cols=4, n_agents=1, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=20,
        seasonal=SeasonalConfig(season_period=4),
    ))
    env.reset(seed=0)
    assert env.season == 0  # Spring
    for _ in range(2):
        env.step(int(Action.NORTH))
    assert env.season == 2  # Autumn (phase=0.5 after 2 ticks on period 4)


def test_solo_seasonal_terminates_on_death() -> None:
    env = SoloSeasonalEnv(SeasonalMultiAgentConfig(
        rows=4, cols=4, n_agents=1, max_energy=2.0, start_energy=1.0,
        metabolism=1.0, food_value=2.0, death_penalty=5.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=10,
        seasonal=SeasonalConfig(season_period=20),
    ))
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(int(Action.NORTH))
    assert terminated is True
    assert reward < -5.0  # step_reward + death_penalty


def test_solo_seasonal_seed_reproducibility() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=6, cols=6, n_agents=1, initial_food_density=0.1,
        food_respawn_lambda=0.0, max_steps=10,
    )
    env1 = SoloSeasonalEnv(cfg)
    env2 = SoloSeasonalEnv(cfg)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
