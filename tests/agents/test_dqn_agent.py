"""Tests pour le wrapper DQNAgent AetherLife."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("torch", reason="suite complete : requiert torch")
pytest.importorskip("mw_ia", reason="suite complete : requiert le repo sibling MW_IA")

from mw_ia.config import DQNConfig

from aetherlife.agents.dqn_agent import DQNAgent
from aetherlife.config import FoodGridConfig
from aetherlife.world.food_grid import FoodGrid


@pytest.fixture
def tiny_env() -> FoodGrid:
    cfg = FoodGridConfig(rows=4, cols=4, initial_food_density=0.0, max_steps=10)
    return FoodGrid(cfg)


@pytest.fixture
def tiny_dqn_cfg() -> DQNConfig:
    return DQNConfig(
        hidden_layers=(16,), batch_size=4, replay_capacity=64,
        min_replay_to_learn=4, target_sync_steps=10, use_amp=False,
    )


def test_dqn_agent_constructs_on_cpu(tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    assert agent.global_step == 0


def test_dqn_agent_act_returns_valid_action(
    tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    obs, _ = tiny_env.reset(seed=0)
    a = agent.act(obs)
    assert 0 <= a < 4


def test_dqn_agent_greedy_act_no_random(
    tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    obs, _ = tiny_env.reset(seed=0)
    a1 = agent.act(obs, greedy=True)
    a2 = agent.act(obs, greedy=True)
    assert a1 == a2


def test_dqn_agent_observe_increments_step(
    tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    obs, _ = tiny_env.reset(seed=0)
    a = agent.act(obs)
    next_obs, r, term, trunc, _ = tiny_env.step(a)
    agent.observe(obs, a, r, next_obs, term or trunc)
    assert agent.global_step == 1


def test_dqn_agent_save_load_roundtrip(
    tmp_path: Path, tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    obs, _ = tiny_env.reset(seed=0)
    for _ in range(5):
        a = agent.act(obs)
        next_obs, r, term, trunc, _ = tiny_env.step(a)
        agent.observe(obs, a, r, next_obs, term or trunc)
        obs = next_obs
        if term or trunc:
            obs, _ = tiny_env.reset(seed=0)

    ckpt = tmp_path / "dqn.pt"
    agent.save(ckpt)
    assert ckpt.exists()

    agent2 = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=99)
    agent2.load(ckpt)
    obs, _ = tiny_env.reset(seed=0)
    np.testing.assert_array_equal(
        np.array([agent.act(obs, greedy=True)]),
        np.array([agent2.act(obs, greedy=True)]),
    )
