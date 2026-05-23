"""Tests pour IndependentDQNAgent (V2 shared-weights IDQN)."""
from __future__ import annotations

from pathlib import Path

import pytest
from mw_ia.config import DQNConfig

from aetherlife.agents.independent_dqn import IndependentDQNAgent
from aetherlife.world.multi_agent_grid import MultiAgentFoodGrid, MultiAgentForagerConfig


@pytest.fixture
def tiny_ma_env() -> MultiAgentFoodGrid:
    cfg = MultiAgentForagerConfig(
        rows=5, cols=5, n_agents=3, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, initial_food_density=0.1,
        food_respawn_lambda=0.0, max_steps=15,
    )
    return MultiAgentFoodGrid(cfg)


@pytest.fixture
def tiny_dqn_cfg() -> DQNConfig:
    return DQNConfig(
        hidden_layers=(16,), batch_size=8, replay_capacity=128,
        min_replay_to_learn=8, target_sync_steps=20, use_amp=False,
    )


def test_idqn_constructs(tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    assert agent.global_step == 0


def test_act_dict_returns_action_per_agent(
    tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    obs_dict, _ = tiny_ma_env.reset(seed=0)
    actions = agent.act_dict(obs_dict)
    assert set(actions.keys()) == {0, 1, 2}
    for a in actions.values():
        assert 0 <= a < 4


def test_observe_dict_increments_step(
    tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    obs_dict, _ = tiny_ma_env.reset(seed=0)
    actions = agent.act_dict(obs_dict)
    next_obs, rewards, terminated, truncated, _ = tiny_ma_env.step(actions)
    dones = {aid: terminated.get(aid, False) or truncated.get(aid, False)
             for aid in actions}
    agent.observe_dict(obs_dict, actions, rewards, next_obs, dones)
    assert agent.global_step == 3  # 3 transitions poussées en 1 tick


def test_idqn_greedy_deterministic(
    tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    obs_dict, _ = tiny_ma_env.reset(seed=0)
    a1 = agent.act_dict(obs_dict, greedy=True)
    a2 = agent.act_dict(obs_dict, greedy=True)
    assert a1 == a2


def test_idqn_save_load(
    tmp_path: Path, tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    ckpt = tmp_path / "idqn.pt"
    agent.save(ckpt)
    assert ckpt.exists()
    agent2 = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=99)
    agent2.load(ckpt)
