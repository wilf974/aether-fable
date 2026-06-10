"""Tests pour le runner DRQN V3.5."""
from __future__ import annotations

from pathlib import Path

import pytest
pytest.importorskip("torch", reason="suite complete : requiert torch")
pytest.importorskip("mw_ia", reason="suite complete : requiert le repo sibling MW_IA")

from mw_ia.config import DRQNConfig

from aetherlife.agents.recurrent_dqn_agent import RecurrentDQNAgent
from aetherlife.training.recurrent_dqn_runner import (
    DRQNAssessmentMetric,
    DRQNEpisodeMetric,
    DRQNRunnerResult,
    drqn_assess,
    run_drqn_training,
)
from aetherlife.world.seasonal_grid import SeasonalConfig, SeasonalMultiAgentConfig
from aetherlife.world.seasonal_solo_env import SoloSeasonalEnv


@pytest.fixture
def tiny_env() -> SoloSeasonalEnv:
    cfg = SeasonalMultiAgentConfig(
        rows=5, cols=5, n_agents=1, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=3.0,
        initial_food_density=0.1, food_respawn_lambda=0.5, max_steps=20,
        seasonal=SeasonalConfig(season_period=20),
    )
    return SoloSeasonalEnv(cfg)


@pytest.fixture
def tiny_drqn_cfg() -> DRQNConfig:
    return DRQNConfig(
        fc_hidden=16, lstm_hidden=16, sequence_length=8,
        batch_size=4, replay_capacity=64, min_episodes_to_learn=2,
        target_sync_steps=50, train_steps_per_episode=1,
        max_steps_per_episode=20, use_amp=False,
        epsilon_decay_steps=200,
    )


def test_drqn_assess_returns_metric(
    tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    m = drqn_assess(tiny_env, agent, n_episodes=2)
    assert isinstance(m, DRQNAssessmentMetric)
    assert 0 <= m.survival_rate <= 1.0


def test_drqn_run_training_completes(
    tmp_path: Path, tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    result = run_drqn_training(
        tiny_env, agent,
        n_episodes=6, assess_every=3, assess_episodes=2,
        checkpoint_path=tmp_path / "drqn_best.pt",
        patience=5,
    )
    assert isinstance(result, DRQNRunnerResult)
    assert len(result.train_metrics) == 6
    assert len(result.assessment_metrics) == 2
    for m in result.train_metrics:
        assert isinstance(m, DRQNEpisodeMetric)


def test_drqn_run_saves_checkpoint(
    tmp_path: Path, tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    ckpt = tmp_path / "drqn_best.pt"
    run_drqn_training(
        tiny_env, agent,
        n_episodes=3, assess_every=3, assess_episodes=2,
        checkpoint_path=ckpt,
    )
    assert ckpt.exists()


def test_drqn_callbacks_fire(
    tmp_path: Path, tiny_env: SoloSeasonalEnv, tiny_drqn_cfg: DRQNConfig
) -> None:
    agent = RecurrentDQNAgent(
        obs_dim=tiny_env.n_states, n_actions=tiny_env.n_actions,
        cfg=tiny_drqn_cfg, device="cpu", seed=0,
    )
    eps: list[DRQNEpisodeMetric] = []
    asses: list[tuple[DRQNAssessmentMetric, bool]] = []
    run_drqn_training(
        tiny_env, agent,
        n_episodes=6, assess_every=3, assess_episodes=2,
        checkpoint_path=tmp_path / "drqn_best.pt",
        on_episode_end=lambda m: eps.append(m),
        on_assess=lambda m, imp: asses.append((m, imp)),
    )
    assert len(eps) == 6
    assert len(asses) == 2
    assert asses[0][1] is True
