"""Tests pour DQN runner — smoke entraînement court + assessment."""
from __future__ import annotations

from pathlib import Path

import pytest
pytest.importorskip("torch", reason="suite complete : requiert torch")
pytest.importorskip("mw_ia", reason="suite complete : requiert le repo sibling MW_IA")

from mw_ia.config import DQNConfig

from aetherlife.agents.dqn_agent import DQNAgent
from aetherlife.config import FoodGridConfig
from aetherlife.training.dqn_runner import (
    AssessmentMetric,
    EpisodeMetric,
    DQNRunnerResult,
    assess,
    run_dqn_training,
)
from aetherlife.world.food_grid import FoodGrid


@pytest.fixture
def tiny_env() -> FoodGrid:
    cfg = FoodGridConfig(
        rows=5, cols=5, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=5.0,
        initial_food_density=0.1, food_respawn_lambda=0.3, max_steps=30,
        start_position=(2, 2),
    )
    return FoodGrid(cfg)


@pytest.fixture
def tiny_dqn_cfg() -> DQNConfig:
    return DQNConfig(
        hidden_layers=(32,),
        episodes=20,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=500,
        gamma=0.95,
        lr=1e-3,
        batch_size=16,
        replay_capacity=500,
        min_replay_to_learn=32,
        target_sync_steps=50,
        train_every=4,
        use_amp=False,
    )


def test_assess_returns_metric(tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    m = assess(tiny_env, agent, n_episodes=3)
    assert isinstance(m, AssessmentMetric)
    assert 0.0 <= m.survival_rate <= 1.0
    assert m.mean_lifespan > 0


def test_run_dqn_training_returns_result(
    tmp_path: Path, tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    result = run_dqn_training(
        tiny_env, agent,
        n_episodes=10, assess_every=5, assess_episodes=2,
        checkpoint_path=tmp_path / "best.pt",
        patience=5,
    )
    assert isinstance(result, DQNRunnerResult)
    assert len(result.train_metrics) == 10
    assert len(result.assessment_metrics) == 2
    for m in result.train_metrics:
        assert isinstance(m, EpisodeMetric)
    assert result.final_episode == 9


def test_training_creates_checkpoint(
    tmp_path: Path, tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    ckpt = tmp_path / "best.pt"
    run_dqn_training(
        tiny_env, agent,
        n_episodes=5, assess_every=5, assess_episodes=2,
        checkpoint_path=ckpt,
    )
    assert ckpt.exists()


def test_training_callbacks_fire(
    tmp_path: Path, tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    ep_calls: list[EpisodeMetric] = []
    eval_calls: list[tuple[AssessmentMetric, bool]] = []
    run_dqn_training(
        tiny_env, agent,
        n_episodes=6, assess_every=3, assess_episodes=2,
        checkpoint_path=tmp_path / "best.pt",
        on_episode_end=lambda m: ep_calls.append(m),
        on_assess=lambda m, imp: eval_calls.append((m, imp)),
    )
    assert len(ep_calls) == 6
    assert len(eval_calls) == 2
    assert eval_calls[0][1] is True  # premier assess = nouveau best


def test_load_after_train_recovers_weights(
    tmp_path: Path, tiny_env: FoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent1 = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=0)
    ckpt = tmp_path / "best.pt"
    run_dqn_training(
        tiny_env, agent1,
        n_episodes=5, assess_every=5, assess_episodes=2,
        checkpoint_path=ckpt,
    )
    agent2 = DQNAgent(tiny_env, tiny_dqn_cfg, device="cpu", seed=99)
    agent2.load(ckpt)
    assert ckpt.exists()
