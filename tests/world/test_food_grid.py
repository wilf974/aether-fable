"""Tests pour FoodGrid (V1 core env)."""
from __future__ import annotations

import numpy as np
import pytest

from aetherlife.config import FoodGridConfig
from aetherlife.world.food_grid import Action, FoodGrid


@pytest.fixture
def small_cfg() -> FoodGridConfig:
    """Config 5x5, peu de food, pas de respawn, max_energy bornée pour les tests."""
    return FoodGridConfig(
        rows=5,
        cols=5,
        max_energy=10.0,
        start_energy=5.0,
        metabolism=1.0,
        food_value=3.0,
        death_penalty=2.0,
        initial_food_density=0.0,
        food_respawn_lambda=0.0,
        max_steps=20,
        start_position=(2, 2),
    )


def test_reset_returns_valid_obs(small_cfg: FoodGridConfig) -> None:
    env = FoodGrid(small_cfg)
    obs, info = env.reset(seed=0)
    assert obs.shape == (small_cfg.obs_dim,)
    assert obs.dtype == np.float32
    assert env.pos == (2, 2)
    assert env.energy == 5.0
    assert env.step_count == 0
    assert info["step"] == 0


def test_step_decreases_energy_by_metabolism(small_cfg: FoodGridConfig) -> None:
    env = FoodGrid(small_cfg)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(Action.NORTH)
    assert env.energy == 4.0  # 5 - 1
    assert reward == -1.0
    assert not terminated
    assert not truncated
    assert info["ate"] is False


def test_step_eats_food_when_on_food_cell(small_cfg: FoodGridConfig) -> None:
    env = FoodGrid(small_cfg)
    env.reset(seed=0)
    env._food_mask[1, 2] = True  # noqa: SLF001 — test internal state
    obs, reward, terminated, truncated, info = env.step(Action.NORTH)
    assert env.pos == (1, 2)
    assert info["ate"] is True
    assert env.energy == 5.0 - 1.0 + 3.0  # = 7
    assert reward == 3.0 - 1.0  # = 2
    assert env.food_count == 0


def test_terminated_when_energy_zero(small_cfg: FoodGridConfig) -> None:
    env = FoodGrid(small_cfg)
    env.reset(seed=0)
    env._energy = 1.0  # noqa: SLF001
    obs, reward, terminated, truncated, info = env.step(Action.NORTH)
    assert env.energy == 0.0
    assert terminated
    assert reward == -1.0 - 2.0  # step_reward + death_penalty


def test_truncated_at_max_steps() -> None:
    cfg = FoodGridConfig(
        rows=5,
        cols=5,
        max_energy=1000.0,
        start_energy=500.0,
        metabolism=0.01,
        food_value=1.0,
        initial_food_density=0.0,
        food_respawn_lambda=0.0,
        max_steps=3,
        start_position=(2, 2),
    )
    env = FoodGrid(cfg)
    env.reset(seed=0)
    env.step(Action.NORTH)
    env.step(Action.NORTH)
    obs, reward, terminated, truncated, info = env.step(Action.NORTH)
    assert truncated
    assert not terminated


def test_action_clamp_at_borders(small_cfg: FoodGridConfig) -> None:
    env = FoodGrid(small_cfg)
    env.reset(seed=0)
    for _ in range(10):
        env.step(Action.NORTH)
    assert env.pos[0] == 0
    for _ in range(10):
        env.step(Action.WEST)
    assert env.pos == (0, 0)


def test_food_respawn_increases_count() -> None:
    cfg = FoodGridConfig(
        rows=10,
        cols=10,
        initial_food_density=0.0,
        food_respawn_lambda=5.0,
        max_steps=100,
    )
    env = FoodGrid(cfg)
    env.reset(seed=0)
    assert env.food_count == 0
    env.step(Action.NORTH)
    assert env.food_count > 0


def test_seed_reproducibility() -> None:
    cfg = FoodGridConfig(rows=8, cols=8, initial_food_density=0.2)
    env1 = FoodGrid(cfg)
    env2 = FoodGrid(cfg)
    env1.reset(seed=42)
    env2.reset(seed=42)
    assert np.array_equal(env1.food_mask, env2.food_mask)


def test_seed_different_layouts() -> None:
    cfg = FoodGridConfig(rows=8, cols=8, initial_food_density=0.3)
    env1 = FoodGrid(cfg)
    env2 = FoodGrid(cfg)
    env1.reset(seed=1)
    env2.reset(seed=2)
    assert not np.array_equal(env1.food_mask, env2.food_mask)


def test_initial_food_count_respects_density() -> None:
    cfg = FoodGridConfig(rows=10, cols=10, initial_food_density=0.1)
    env = FoodGrid(cfg)
    env.reset(seed=0)
    assert env.food_count == 10  # 0.1 * 100


def test_start_position_never_has_food() -> None:
    cfg = FoodGridConfig(rows=4, cols=4, initial_food_density=0.9, start_position=(0, 0))
    env = FoodGrid(cfg)
    env.reset(seed=0)
    assert not env.food_mask[0, 0]


def test_observation_one_hot_position(small_cfg: FoodGridConfig) -> None:
    env = FoodGrid(small_cfg)
    obs, _ = env.reset(seed=0)
    n_cells = small_cfg.rows * small_cfg.cols
    pos_one_hot = obs[:n_cells]
    assert pos_one_hot.sum() == 1.0
    expected_idx = 2 * small_cfg.cols + 2  # (2, 2)
    assert pos_one_hot[expected_idx] == 1.0


def test_observation_energy_normalized(small_cfg: FoodGridConfig) -> None:
    env = FoodGrid(small_cfg)
    obs, _ = env.reset(seed=0)
    assert obs[-1] == pytest.approx(5.0 / 10.0)
    env.step(Action.NORTH)
    obs2 = env._observation()  # noqa: SLF001
    assert obs2[-1] == pytest.approx(4.0 / 10.0)


def test_render_ascii_shows_agent_and_food(small_cfg: FoodGridConfig) -> None:
    env = FoodGrid(small_cfg)
    env.reset(seed=0)
    env._food_mask[0, 0] = True  # noqa: SLF001
    text = env.render_ascii()
    assert "A" in text
    assert "*" in text
    assert "energy" in text
