"""Tests pour le runner ConvDQN V3.6."""
from __future__ import annotations

from pathlib import Path

import pytest
from mw_ia.config import ConvDQNConfig

from aetherlife.agents.conv_dqn_agent import ConvDQNAgent
from aetherlife.training.conv_dqn_runner import (
    ConvAssessmentMetric,
    ConvDQNRunnerResult,
    ConvEpisodeMetric,
    conv_assess,
    run_conv_dqn_training,
)
from aetherlife.world.seasonal_grid import SeasonalConfig, SeasonalMultiAgentConfig
from aetherlife.world.seasonal_solo_env import SoloSeasonalEnv


@pytest.fixture
def tiny_env() -> SoloSeasonalEnv:
    cfg = SeasonalMultiAgentConfig(
        rows=4, cols=4, n_agents=1, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=3.0,
        initial_food_density=0.15, food_respawn_lambda=0.5, max_steps=15,
        seasonal=SeasonalConfig(season_period=15),
    )
    return SoloSeasonalEnv(cfg)


@pytest.fixture
def tiny_conv_cfg() -> ConvDQNConfig:
    return ConvDQNConfig(
        conv_channels=(8,), kernel_size=3, padding=1, fc_hidden=16,
        batch_size=4, replay_capacity=64, min_replay_to_learn=4,
        target_sync_steps=20, train_every=1, use_amp=False,
        epsilon_decay_steps=200, double_dqn=True,
    )


def test_conv_assess_returns_metric(
    tiny_env: SoloSeasonalEnv, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = ConvDQNAgent(4, 4, 4, 4, cfg=tiny_conv_cfg, device="cpu", seed=0)
    m = conv_assess(tiny_env, agent, n_episodes=2)
    assert isinstance(m, ConvAssessmentMetric)
    assert 0 <= m.survival_rate <= 1.0


def test_conv_run_training(
    tmp_path: Path, tiny_env: SoloSeasonalEnv, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = ConvDQNAgent(4, 4, 4, 4, cfg=tiny_conv_cfg, device="cpu", seed=0)
    result = run_conv_dqn_training(
        tiny_env, agent,
        n_episodes=6, assess_every=3, assess_episodes=2,
        checkpoint_path=tmp_path / "conv_best.pt",
        patience=5,
    )
    assert isinstance(result, ConvDQNRunnerResult)
    assert len(result.train_metrics) == 6
    assert len(result.assessment_metrics) == 2


def test_conv_run_saves_checkpoint(
    tmp_path: Path, tiny_env: SoloSeasonalEnv, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = ConvDQNAgent(4, 4, 4, 4, cfg=tiny_conv_cfg, device="cpu", seed=0)
    ckpt = tmp_path / "conv_best.pt"
    run_conv_dqn_training(
        tiny_env, agent,
        n_episodes=3, assess_every=3, assess_episodes=2,
        checkpoint_path=ckpt,
    )
    assert ckpt.exists()
