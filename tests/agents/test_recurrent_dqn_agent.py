"""Tests pour le wrapper RecurrentDQNAgent AetherLife."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mw_ia.config import DRQNConfig

from aetherlife.agents.recurrent_dqn_agent import RecurrentDQNAgent
from aetherlife.world.seasonal_grid import SeasonalConfig, SeasonalMultiAgentConfig
from aetherlife.world.seasonal_solo_env import SoloSeasonalEnv


@pytest.fixture
def tiny_env() -> SoloSeasonalEnv:
    cfg = SeasonalMultiAgentConfig(
        rows=4, cols=4, n_agents=1, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=10,
        seasonal=SeasonalConfig(season_period=20),
    )
    return SoloSeasonalEnv(cfg)


@pytest.fixture
def tiny_drqn_cfg() -> DRQNConfig:
    return DRQNConfig(
        fc_hidden=16, lstm_hidden=16, sequence_length=8,
        batch_size=4, replay_capacity=64, min_episodes_to_learn=4,
        target_sync_steps=50, train_steps_per_episode=1,
        max_steps_per_episode=10, use_amp=False,
    )


def test_drqn_agent_constructs(
    tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    assert agent.global_step == 0


def test_drqn_act_returns_valid_action(
    tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    agent.reset_hidden()
    agent.begin_episode()
    obs, _ = tiny_env.reset(seed=0)
    a = agent.act(obs)
    assert 0 <= a < 4


def test_drqn_hidden_state_maintained_across_steps(
    tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    agent.reset_hidden()
    agent.begin_episode()
    obs, _ = tiny_env.reset(seed=0)
    # Hidden state initial None
    assert agent._impl._hidden_state is None  # noqa: SLF001
    agent.act(obs, greedy=True)
    # Hidden state non-None après 1 forward
    assert agent._impl._hidden_state is not None  # noqa: SLF001


def test_drqn_end_episode_pushes_trajectory(
    tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    agent.reset_hidden()
    agent.begin_episode()
    obs, _ = tiny_env.reset(seed=0)
    for _ in range(3):
        a = agent.act(obs)
        next_obs, r, term, trunc, _ = tiny_env.step(a)
        agent.observe(obs, a, r, next_obs, term or trunc)
        obs = next_obs
        if term or trunc:
            break
    assert len(agent._impl._episode_trajectory) > 0  # noqa: SLF001
    agent.end_episode()
    assert len(agent._impl.buffer) == 1


def test_drqn_save_load(
    tmp_path: Path, tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    ckpt = tmp_path / "drqn.pt"
    agent.save(ckpt)
    assert ckpt.exists()
    agent2 = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=99,
    )
    agent2.load(ckpt)
