"""Tests pour le MA runner V2."""
from __future__ import annotations

from pathlib import Path

import pytest
from mw_ia.config import DQNConfig

from aetherlife.agents.independent_dqn import IndependentDQNAgent
from aetherlife.training.multi_agent_runner import (
    MAAssessmentMetric,
    MAEpisodeMetric,
    MARunnerResult,
    ma_assess,
    run_ma_training,
)
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)


@pytest.fixture
def tiny_ma_env() -> MultiAgentFoodGrid:
    cfg = MultiAgentForagerConfig(
        rows=5, cols=5, n_agents=3, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=3.0,
        initial_food_density=0.15, food_respawn_lambda=0.5, max_steps=20,
    )
    return MultiAgentFoodGrid(cfg)


@pytest.fixture
def tiny_dqn_cfg() -> DQNConfig:
    return DQNConfig(
        hidden_layers=(16,), batch_size=8, replay_capacity=128,
        min_replay_to_learn=8, target_sync_steps=20, train_every=2, use_amp=False,
    )


def test_ma_assess_returns_metric(
    tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    m = ma_assess(tiny_ma_env, agent, n_episodes=2)
    assert isinstance(m, MAAssessmentMetric)
    assert 0.0 <= m.mean_alive_rate <= 1.0


def test_run_ma_training_returns_result(
    tmp_path: Path, tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    result = run_ma_training(
        tiny_ma_env, agent,
        n_episodes=6, assess_every=3, assess_episodes=2,
        checkpoint_path=tmp_path / "ma_best.pt",
        patience=5,
    )
    assert isinstance(result, MARunnerResult)
    assert len(result.train_metrics) == 6
    assert len(result.assessment_metrics) == 2
    for m in result.train_metrics:
        assert isinstance(m, MAEpisodeMetric)


def test_ma_training_saves_checkpoint(
    tmp_path: Path, tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    ckpt = tmp_path / "ma_best.pt"
    run_ma_training(
        tiny_ma_env, agent,
        n_episodes=3, assess_every=3, assess_episodes=2,
        checkpoint_path=ckpt,
    )
    assert ckpt.exists()
