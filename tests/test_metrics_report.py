"""Tests scripts/metrics_report.py — lecture/résumé des runs télémétrie."""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import pytest

from aetherlife.telemetry import MetricsLogger
from metrics_report import load_run, main, numeric_series


@pytest.fixture
def run_dir(tmp_path):
    with MetricsLogger(tmp_path, run_id="t", config={"seed": 3, "regime": "x"}) as ml:
        ml.log(100, alive=20, mean_loss=0.5, phase="train")
        ml.log(200, alive=18, mean_loss=0.3, phase="train")
        ml.log(200, policy_divergence=0.12)
        ml.summary(final_alive=18)
    return tmp_path


def test_load_run(run_dir):
    run = load_run(run_dir)
    assert len(run["records"]) == 3
    assert run["summary"]["final_alive"] == 18
    assert run["config"]["seed"] == 3


def test_load_run_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)


def test_numeric_series(run_dir):
    series = numeric_series(load_run(run_dir)["records"])
    assert series["alive"] == [(100, 20.0), (200, 18.0)]
    assert series["policy_divergence"] == [(200, 0.12)]
    assert "run_id" not in series and "phase" not in series


def test_main_prints_report(run_dir, capsys):
    assert main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "run_id=t" in out
    assert "alive" in out
    assert "final_alive" in out


def test_main_no_valid_dir(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
