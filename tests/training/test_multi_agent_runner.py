"""Tests pour le MA runner V2."""
from __future__ import annotations

from pathlib import Path

import pytest
pytest.importorskip("torch", reason="suite complete : requiert torch")
pytest.importorskip("mw_ia", reason="suite complete : requiert le repo sibling MW_IA")

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


def test_run_ma_training_writes_metrics_jsonl(
    tmp_path: Path, tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    """V2.5 — metrics_dir produit metrics.jsonl + run_summary.json."""
    import json

    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    mdir = tmp_path / "telemetry"
    run_ma_training(
        tiny_ma_env, agent,
        n_episodes=6, assess_every=3, assess_episodes=2,
        checkpoint_path=tmp_path / "ma_best.pt",
        metrics_dir=mdir,
    )
    recs = [json.loads(l) for l in (mdir / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in recs if r["phase"] == "train"]
    assess = [r for r in recs if r["phase"] == "assess"]
    assert len(train) == 6
    assert len(assess) == 2
    assert "mean_lifespan" in train[0]
    assert "improved" in assess[0]
    summ = json.loads((mdir / "run_summary.json").read_text())
    assert "best_assessment_score" in summ


def test_run_ma_training_no_metrics_dir_writes_nothing(
    tmp_path: Path, tiny_ma_env: MultiAgentFoodGrid, tiny_dqn_cfg: DQNConfig
) -> None:
    agent = IndependentDQNAgent(tiny_ma_env, tiny_dqn_cfg, device="cpu", seed=0)
    run_ma_training(
        tiny_ma_env, agent,
        n_episodes=2, assess_every=5, assess_episodes=1,
        checkpoint_path=tmp_path / "ma_best.pt",
    )
    assert not (tmp_path / "metrics.jsonl").exists()
