"""Tests pour le MA ConvDQN runner V3.7."""
from __future__ import annotations

from pathlib import Path

import pytest
from mw_ia.config import ConvDQNConfig

from aetherlife.agents.independent_conv_dqn import IndependentConvDQNAgent
from aetherlife.training.multi_agent_conv_runner import (
    MAConvAssessmentMetric,
    MAConvEpisodeMetric,
    MAConvRunnerResult,
    ma_conv_assess,
    run_ma_conv_training,
)
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


@pytest.fixture
def tiny_env() -> SeasonalMultiAgentFoodGrid:
    cfg = SeasonalMultiAgentConfig(
        rows=5, cols=5, n_agents=3, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=3.0,
        initial_food_density=0.15, food_respawn_lambda=0.5, max_steps=15,
        seasonal=SeasonalConfig(season_period=15),
    )
    return SeasonalMultiAgentFoodGrid(cfg)


@pytest.fixture
def tiny_conv_cfg() -> ConvDQNConfig:
    return ConvDQNConfig(
        conv_channels=(8,), kernel_size=3, padding=1, fc_hidden=16,
        batch_size=4, replay_capacity=64, min_replay_to_learn=4,
        target_sync_steps=20, train_every=1, use_amp=False,
        epsilon_decay_steps=200, double_dqn=True,
    )


def test_ma_conv_assess(
    tiny_env: SeasonalMultiAgentFoodGrid, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=0
    )
    m = ma_conv_assess(tiny_env, agent, n_episodes=2)
    assert isinstance(m, MAConvAssessmentMetric)
    assert 0 <= m.mean_alive_rate <= 1.0


def test_ma_conv_run_training(
    tmp_path: Path, tiny_env: SeasonalMultiAgentFoodGrid, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=0
    )
    result = run_ma_conv_training(
        tiny_env, agent,
        n_episodes=4, assess_every=2, assess_episodes=2,
        checkpoint_path=tmp_path / "ma_conv_best.pt",
        patience=5,
    )
    assert isinstance(result, MAConvRunnerResult)
    assert len(result.train_metrics) == 4
    assert len(result.assessment_metrics) == 2
    for m in result.train_metrics:
        assert isinstance(m, MAConvEpisodeMetric)


def test_ma_conv_saves_checkpoint(
    tmp_path: Path, tiny_env: SeasonalMultiAgentFoodGrid, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = IndependentConvDQNAgent(
        tiny_env, tiny_conv_cfg, device="cpu", seed=0
    )
    ckpt = tmp_path / "ma_conv_best.pt"
    run_ma_conv_training(
        tiny_env, agent,
        n_episodes=2, assess_every=2, assess_episodes=2,
        checkpoint_path=ckpt,
    )
    assert ckpt.exists()
