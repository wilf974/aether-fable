"""Tests aetherlife.telemetry — logger structuré + métriques JSONL."""
from __future__ import annotations

import json
import logging

import numpy as np

from aetherlife.telemetry import MetricsLogger, get_logger


def _read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_metrics_logger_roundtrip(tmp_path):
    with MetricsLogger(tmp_path, run_id="test_run") as ml:
        ml.log(100, alive=42, mean_loss=0.123)
        ml.log(200, alive=40, mean_loss=0.101)
    recs = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(recs) == 2
    assert recs[0]["run_id"] == "test_run"
    assert recs[0]["step"] == 100
    assert recs[0]["alive"] == 42
    assert recs[1]["mean_loss"] == 0.101
    assert recs[0]["wall_time"] >= 0


def test_metrics_logger_numpy_types(tmp_path):
    with MetricsLogger(tmp_path) as ml:
        ml.log(1, loss=np.float32(0.5), count=np.int64(7), vec=np.array([1, 2]))
    rec = _read_jsonl(tmp_path / "metrics.jsonl")[0]
    assert rec["loss"] == 0.5
    assert rec["count"] == 7
    assert rec["vec"] == [1, 2]


def test_metrics_logger_crash_safe(tmp_path):
    """Le fichier est lisible AVANT close() — flush à chaque log()."""
    ml = MetricsLogger(tmp_path, run_id="crash")
    ml.log(1, alive=10)
    recs = _read_jsonl(tmp_path / "metrics.jsonl")  # sans close()
    assert recs[0]["alive"] == 10
    ml.close()


def test_metrics_logger_config_and_summary(tmp_path):
    with MetricsLogger(tmp_path, config={"seed": 7, "regime": "coordination"}) as ml:
        ml.summary(final_alive=12, verdict="DIVERGENT")
    cfg = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["seed"] == 7
    summ = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summ["final_alive"] == 12
    assert "duration_s" in summ


def test_metrics_logger_append_mode(tmp_path):
    with MetricsLogger(tmp_path, run_id="a") as ml:
        ml.log(1, x=1)
    with MetricsLogger(tmp_path, run_id="b") as ml:
        ml.log(2, x=2)
    recs = _read_jsonl(tmp_path / "metrics.jsonl")
    assert [r["run_id"] for r in recs] == ["a", "b"]


def test_get_logger_idempotent(tmp_path):
    log_file = tmp_path / "run.log"
    l1 = get_logger("aetherlife.test_idem", log_file=log_file)
    n_handlers = len(l1.handlers)
    l2 = get_logger("aetherlife.test_idem", log_file=log_file)
    assert l2 is l1
    assert len(l2.handlers) == n_handlers  # pas de duplication


def test_get_logger_writes_file(tmp_path):
    log_file = tmp_path / "sub" / "run.log"
    logger = get_logger("aetherlife.test_file", log_file=log_file, level=logging.INFO)
    logger.info("hello telemetry")
    for h in logger.handlers:
        h.flush()
    assert "hello telemetry" in log_file.read_text(encoding="utf-8")
