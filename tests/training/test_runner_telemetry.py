"""Tests V2.5 — intégration MetricsLogger dans run_ma_training, sans torch.

Utilise un agent stub (random valide) : l'env MultiAgentFoodGrid est numpy
pur, donc ce test tourne partout (CI core incluse), contrairement aux tests
IDQN qui requièrent torch/mw_ia.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aetherlife.training.multi_agent_runner import run_ma_training
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)


class _StubAgent:
    """Implémente l'API minimale attendue par run_ma_training."""

    def __init__(self, env: MultiAgentFoodGrid, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_actions = env.n_actions
        self.epsilon = 1.0
        self.last_loss: float | None = None
        self.global_step = 0

    def act_dict(self, obs_dict, greedy: bool = False):
        self.global_step += len(obs_dict)
        return {
            aid: int(self._rng.integers(self._n_actions))
            for aid in obs_dict
        }

    def observe_dict(self, obs, actions, rewards, next_obs, done):
        pass

    def save(self, path) -> None:
        Path(path).write_bytes(b"stub")

    def load(self, path) -> None:
        pass


@pytest.fixture
def tiny_env() -> MultiAgentFoodGrid:
    cfg = MultiAgentForagerConfig(
        rows=5, cols=5, n_agents=3, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=3.0,
        initial_food_density=0.15, food_respawn_lambda=0.5, max_steps=15,
    )
    return MultiAgentFoodGrid(cfg)


def test_metrics_dir_produces_jsonl_and_summary(tmp_path, tiny_env):
    mdir = tmp_path / "telemetry"
    result = run_ma_training(
        tiny_env, _StubAgent(tiny_env),
        n_episodes=6, assess_every=3, assess_episodes=2,
        checkpoint_path=tmp_path / "best.pt",
        metrics_dir=mdir,
    )
    recs = [json.loads(l) for l in (mdir / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in recs if r["phase"] == "train"]
    assess = [r for r in recs if r["phase"] == "assess"]
    assert len(train) == 6
    assert len(assess) == 2
    assert {"step", "wall_time", "mean_lifespan", "total_reward"} <= train[0].keys()
    assert "improved" in assess[0]
    summ = json.loads((mdir / "run_summary.json").read_text())
    assert summ["final_episode"] == result.final_episode
    assert "best_assessment_score" in summ


def test_no_metrics_dir_is_noop(tmp_path, tiny_env):
    run_ma_training(
        tiny_env, _StubAgent(tiny_env),
        n_episodes=2, assess_every=5, assess_episodes=1,
        checkpoint_path=tmp_path / "best.pt",
    )
    assert not (tmp_path / "metrics.jsonl").exists()
    assert not (tmp_path / "run_summary.json").exists()
