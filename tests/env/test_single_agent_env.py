"""Tests pour SoloForagerEnv (wrapper Gymnasium)."""
from __future__ import annotations

import numpy as np

from aetherlife.config import FoodGridConfig
from aetherlife.env.single_agent_env import SoloForagerEnv


def test_observation_space_box_correct_shape() -> None:
    cfg = FoodGridConfig(rows=4, cols=4)
    env = SoloForagerEnv(cfg)
    assert env.observation_space.shape == (cfg.obs_dim,)
    assert env.observation_space.low.min() == 0.0
    assert env.observation_space.high.max() == 1.0


def test_action_space_discrete_4() -> None:
    env = SoloForagerEnv()
    assert env.action_space.n == 4


def test_reset_returns_obs_in_space() -> None:
    env = SoloForagerEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["step"] == 0


def test_step_returns_5tuple() -> None:
    env = SoloForagerEnv()
    env.reset(seed=0)
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_full_episode_under_truncation() -> None:
    cfg = FoodGridConfig(
        rows=5, cols=5, max_energy=1_000_000.0, start_energy=500_000.0,
        metabolism=0.001, food_value=1.0, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=10,
    )
    env = SoloForagerEnv(cfg)
    env.reset(seed=0)
    for i in range(9):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert not truncated
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


def test_check_env_gymnasium_compliance() -> None:
    """Vérifie la conformité Gymnasium via le checker officiel."""
    from gymnasium.utils.env_checker import check_env

    env = SoloForagerEnv(FoodGridConfig(rows=4, cols=4, max_steps=20))
    check_env(env, skip_render_check=True)


def test_render_ansi() -> None:
    env = SoloForagerEnv(FoodGridConfig(rows=4, cols=4))
    env.reset(seed=0)
    text = env.render()
    assert isinstance(text, str)
    assert "A" in text
