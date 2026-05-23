"""Tests pour IndependentConvDQNAgent V3.7."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mw_ia.config import ConvDQNConfig

from aetherlife.agents.independent_conv_dqn import IndependentConvDQNAgent
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


@pytest.fixture
def tiny_env() -> SeasonalMultiAgentFoodGrid:
    cfg = SeasonalMultiAgentConfig(
        rows=5, cols=5, n_agents=3, initial_food_density=0.1,
        food_respawn_lambda=0.0, max_steps=15,
        seasonal=SeasonalConfig(season_period=15),
    )
    return SeasonalMultiAgentFoodGrid(cfg)


@pytest.fixture
def tiny_conv_cfg() -> ConvDQNConfig:
    return ConvDQNConfig(
        conv_channels=(8,), kernel_size=3, padding=1, fc_hidden=16,
        batch_size=4, replay_capacity=64, min_replay_to_learn=4,
        target_sync_steps=20, train_every=1, use_amp=False,
        double_dqn=True,
    )


def test_ma_conv_constructs(
    tiny_env: SeasonalMultiAgentFoodGrid, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=0
    )
    assert agent.global_step == 0


def test_ma_conv_act_dict(
    tiny_env: SeasonalMultiAgentFoodGrid, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=0
    )
    tiny_env.reset(seed=0)
    obs2d = tiny_env.observation_2d_dict()
    assert set(obs2d.keys()) == {0, 1, 2}
    for o in obs2d.values():
        assert o.shape == (4, 5, 5)
    actions = agent.act_dict(obs2d)
    assert set(actions.keys()) == {0, 1, 2}
    for a in actions.values():
        assert 0 <= a < 4


def test_ma_conv_observe_dict_pushes_all(
    tiny_env: SeasonalMultiAgentFoodGrid, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=0
    )
    tiny_env.reset(seed=0)
    obs2d = tiny_env.observation_2d_dict()
    actions = agent.act_dict(obs2d)
    _, rewards, terminated, truncated, _ = tiny_env.step(actions)
    next_obs2d = tiny_env.observation_2d_dict()
    full_next = {aid: next_obs2d.get(aid, obs2d[aid]) for aid in actions}
    dones = {aid: terminated.get(aid, False) or truncated.get(aid, False) for aid in actions}
    agent.observe_dict(obs2d, actions, rewards, full_next, dones)
    assert agent.global_step == 3


def test_ma_conv_greedy_deterministic(
    tiny_env: SeasonalMultiAgentFoodGrid, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=0
    )
    tiny_env.reset(seed=0)
    obs2d = tiny_env.observation_2d_dict()
    a1 = agent.act_dict(obs2d, greedy=True)
    a2 = agent.act_dict(obs2d, greedy=True)
    assert a1 == a2


def test_ma_conv_save_load(
    tmp_path: Path, tiny_env: SeasonalMultiAgentFoodGrid, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=0
    )
    ckpt = tmp_path / "ma_conv.pt"
    agent.save(ckpt)
    assert ckpt.exists()
    agent2 = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=99
    )
    agent2.load(ckpt)


def test_observation_2d_dict_filters_dead_agents(
    tiny_env: SeasonalMultiAgentFoodGrid,
) -> None:
    tiny_env.reset(seed=0)
    tiny_env._agents[1].alive = False  # noqa: SLF001
    obs2d = tiny_env.observation_2d_dict()
    assert set(obs2d.keys()) == {0, 2}
