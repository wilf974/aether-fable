"""Tests pour MultiAgentFoodGrid (V2 core env)."""
from __future__ import annotations

import numpy as np
import pytest

from aetherlife.world.food_grid import Action
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)


@pytest.fixture
def small_ma_cfg() -> MultiAgentForagerConfig:
    return MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=3, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=3.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=30,
    )


def test_reset_creates_n_agents(small_ma_cfg: MultiAgentForagerConfig) -> None:
    env = MultiAgentFoodGrid(small_ma_cfg)
    obs_dict, info_dict = env.reset(seed=0)
    assert env.n_alive == 3
    assert env.n_dead == 0
    assert set(obs_dict.keys()) == {0, 1, 2}
    for obs in obs_dict.values():
        assert obs.shape == (small_ma_cfg.obs_dim,)
        assert obs.dtype == np.float32


def test_step_processes_all_agents(small_ma_cfg: MultiAgentForagerConfig) -> None:
    env = MultiAgentFoodGrid(small_ma_cfg)
    env.reset(seed=0)
    actions = {0: 0, 1: 1, 2: 2}
    obs, rewards, terminated, truncated, infos = env.step(actions)
    assert set(rewards.keys()) == {0, 1, 2}
    for r in rewards.values():
        assert r == -1.0  # -metabolism, pas de food


def test_agent_dies_when_energy_zero() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=2, max_energy=5.0, start_energy=1.0,
        metabolism=1.0, food_value=2.0, death_penalty=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=10,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    actions = {0: 0, 1: 1}
    obs, rewards, terminated, truncated, infos = env.step(actions)
    assert terminated[0] is True
    assert terminated[1] is True
    assert env.n_alive == 0
    assert env.n_dead == 2
    assert rewards[0] == -1.0 - 10.0


def test_dead_agent_not_in_next_step(small_ma_cfg: MultiAgentForagerConfig) -> None:
    env = MultiAgentFoodGrid(small_ma_cfg)
    env.reset(seed=0)
    env._agents[0].energy = 0.5  # noqa: SLF001 — force la mort au prochain step
    env.step({0: 0, 1: 1, 2: 2})
    assert env.n_alive == 2
    obs2, rewards2, _, _, _ = env.step({1: 0, 2: 0})
    assert set(obs2.keys()) == {1, 2}


def test_food_consumption_deterministic_by_id() -> None:
    """2 agents sur la même food : id le plus bas mange en premier (ordre id-asc)."""
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=2, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=5.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=10,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env._agents[0].pos = (0, 0)  # noqa: SLF001
    env._agents[1].pos = (0, 2)  # noqa: SLF001
    env._food_mask[0, 1] = True  # noqa: SLF001
    obs, rewards, _, _, infos = env.step({0: 3, 1: 2})  # 0 EAST, 1 WEST → vers (0,1)
    assert env._agents[0].pos == (0, 1)  # noqa: SLF001
    assert env._agents[1].pos == (0, 1)  # noqa: SLF001
    assert infos[0]["ate"] is True
    assert infos[1]["ate"] is False
    assert env.food_count == 0


def test_alive_agents_returns_alive_ids(small_ma_cfg: MultiAgentForagerConfig) -> None:
    env = MultiAgentFoodGrid(small_ma_cfg)
    env.reset(seed=0)
    assert sorted(env.alive_agent_ids) == [0, 1, 2]
    env._agents[1].alive = False  # noqa: SLF001
    assert sorted(env.alive_agent_ids) == [0, 2]


def test_truncated_at_max_steps() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=2, max_energy=1_000_000.0, start_energy=500_000.0,
        metabolism=0.001, food_value=1.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=2,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0, 1: 1})
    _, _, terminated, truncated, _ = env.step({0: 0, 1: 1})
    assert truncated[0] is True
    assert truncated[1] is True


def test_seed_reproducibility(small_ma_cfg: MultiAgentForagerConfig) -> None:
    env1 = MultiAgentFoodGrid(small_ma_cfg)
    env2 = MultiAgentFoodGrid(small_ma_cfg)
    env1.reset(seed=42)
    env2.reset(seed=42)
    assert env1.food_count == env2.food_count
    for i in range(3):
        assert env1._agents[i].pos == env2._agents[i].pos  # noqa: SLF001


def test_observation_includes_other_agents_positions() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=3, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=10,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    n_cells = cfg.rows * cfg.cols
    obs_0 = env._observation_for(0)  # noqa: SLF001
    # canal 1 (others) doit avoir exactement 2 cellules à 1.0
    others_channel = obs_0[n_cells : 2 * n_cells]
    assert others_channel.sum() == 2.0  # agents 1 et 2


def test_food_respawn_increases_count() -> None:
    cfg = MultiAgentForagerConfig(
        rows=10, cols=10, n_agents=3, initial_food_density=0.0,
        food_respawn_lambda=5.0, max_steps=10,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    assert env.food_count == 0
    env.step({0: 0, 1: 0, 2: 0})
    assert env.food_count > 0


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        MultiAgentForagerConfig(n_agents=0)
    with pytest.raises(ValueError):
        MultiAgentForagerConfig(n_agents=100, rows=4, cols=4)  # n_agents > rows*cols
