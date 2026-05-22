"""Tests pour BestCheckpointTracker."""
from __future__ import annotations

from pathlib import Path

import pytest

from aetherlife.training.best_checkpoint import BestCheckpointTracker


class _FakeModel:
    def __init__(self) -> None:
        self.save_calls: list[Path] = []
        self.load_calls: list[Path] = []
        self.snapshot: int = 0

    def save(self, path: str | Path) -> None:
        self.save_calls.append(Path(path))

    def load(self, path: str | Path) -> None:
        self.load_calls.append(Path(path))


def test_first_report_always_improves(tmp_path: Path) -> None:
    model = _FakeModel()
    tr = BestCheckpointTracker(save_path=tmp_path / "best.pt", patience=3)
    improved = tr.report(0, 0.5, model)
    assert improved is True
    assert tr.best_score == 0.5
    assert tr.best_step == 0
    assert len(model.save_calls) == 1


def test_no_save_when_score_decreases(tmp_path: Path) -> None:
    model = _FakeModel()
    tr = BestCheckpointTracker(save_path=tmp_path / "best.pt", patience=3)
    tr.report(0, 0.5, model)
    improved = tr.report(1, 0.4, model)
    assert improved is False
    assert tr.best_score == 0.5
    assert len(model.save_calls) == 1
    assert tr.evals_since_best == 1


def test_save_when_score_improves(tmp_path: Path) -> None:
    model = _FakeModel()
    tr = BestCheckpointTracker(save_path=tmp_path / "best.pt", patience=3)
    tr.report(0, 0.5, model)
    improved = tr.report(1, 0.7, model)
    assert improved is True
    assert tr.best_score == 0.7
    assert tr.best_step == 1
    assert len(model.save_calls) == 2
    assert tr.evals_since_best == 0


def test_should_stop_after_patience(tmp_path: Path) -> None:
    model = _FakeModel()
    tr = BestCheckpointTracker(save_path=tmp_path / "best.pt", patience=3)
    tr.report(0, 0.5, model)
    assert tr.should_stop is False
    tr.report(1, 0.4, model)
    tr.report(2, 0.4, model)
    assert tr.should_stop is False
    tr.report(3, 0.4, model)
    assert tr.should_stop is True


def test_min_delta_rejects_marginal_improvement(tmp_path: Path) -> None:
    model = _FakeModel()
    tr = BestCheckpointTracker(save_path=tmp_path / "best.pt", patience=3, min_delta=0.05)
    tr.report(0, 0.50, model)
    improved = tr.report(1, 0.51, model)  # Delta 0.01 < min_delta 0.05
    assert improved is False
    assert tr.best_score == 0.50


def test_rollback_loads_model(tmp_path: Path) -> None:
    model = _FakeModel()
    tr = BestCheckpointTracker(save_path=tmp_path / "best.pt", patience=3)
    tr.report(0, 0.5, model)
    tr.save_path.touch()
    tr.rollback(model)
    assert len(model.load_calls) == 1


def test_rollback_without_checkpoint_raises(tmp_path: Path) -> None:
    model = _FakeModel()
    tr = BestCheckpointTracker(save_path=tmp_path / "no_such.pt", patience=3)
    with pytest.raises(FileNotFoundError):
        tr.rollback(model)


def test_history_records_all_evals(tmp_path: Path) -> None:
    model = _FakeModel()
    tr = BestCheckpointTracker(save_path=tmp_path / "best.pt")
    tr.report(0, 0.5, model)
    tr.report(1, 0.4, model)
    tr.report(2, 0.6, model)
    assert tr.history == [(0, 0.5), (1, 0.4), (2, 0.6)]
