"""Tests aetherlife.analysis.aggregate."""
from __future__ import annotations

import json

from aetherlife.analysis.aggregate import (
    aggregate_metric, collect_runs, get_path, load_run,
)


def _write_run(d, seed, alive, shannon=None, ext=False):
    d.mkdir(parents=True, exist_ok=True)
    report = {
        "final_state": {"n_alive": alive},
        "ecology_v25": {"shannon_diversity": shannon} if shannon is not None else {},
        "extinct": ext,
    }
    (d / f"overnight_v8b1_seed{seed}.json").write_text(json.dumps(report))


def test_get_path_nested_and_list():
    data = {"a": {"b": [10, {"c": 42}]}}
    assert get_path(data, "a.b.1.c") == 42
    assert get_path(data, "a.b.0") == 10
    assert get_path(data, "a.x", default="?") == "?"


def test_load_run_merges(tmp_path):
    d = tmp_path / "run1"
    d.mkdir()
    (d / "run_summary.json").write_text(json.dumps({"final_alive": 5, "x": 1}))
    (d / "overnight_v8b1_seed3.json").write_text(json.dumps({"final_state": {"n_alive": 5}}))
    run = load_run(d)
    assert run["x"] == 1
    assert run["final_state"]["n_alive"] == 5


def test_collect_runs_recursive(tmp_path):
    _write_run(tmp_path / "k1" / "seed1", 1, 80)
    _write_run(tmp_path / "k1" / "seed2", 2, 0, ext=True)
    _write_run(tmp_path / "k4" / "seed1", 1, 100)
    runs = collect_runs(tmp_path)
    assert len(runs) == 3


def test_aggregate_metric_mean(tmp_path):
    _write_run(tmp_path / "s1", 1, 80, shannon=1.2)
    _write_run(tmp_path / "s2", 2, 60, shannon=0.8)
    runs = collect_runs(tmp_path)
    summ = aggregate_metric(runs, "ecology_v25.shannon_diversity")
    assert summ.n == 2
    assert summ.mean == 1.0


def test_aggregate_metric_bool_as_rate(tmp_path):
    _write_run(tmp_path / "s1", 1, 0, ext=True)
    _write_run(tmp_path / "s2", 2, 80, ext=False)
    _write_run(tmp_path / "s3", 3, 0, ext=True)
    runs = collect_runs(tmp_path)
    summ = aggregate_metric(runs, "extinct")
    assert summ.n == 3
    assert round(summ.mean, 4) == round(2 / 3, 4)


def test_aggregate_metric_extinct_derived_from_n_alive(tmp_path):
    """Les vrais reports overnight n'ont PAS de champ 'extinct' : il est dérivé
    de final_state.n_alive == 0 (cf. notes des specs préreg C2)."""
    d1 = tmp_path / "s1"
    d1.mkdir()
    (d1 / "overnight_v8b1_seed1.json").write_text(
        json.dumps({"final_state": {"n_alive": 0}}))
    d2 = tmp_path / "s2"
    d2.mkdir()
    (d2 / "overnight_v8b1_seed2.json").write_text(
        json.dumps({"final_state": {"n_alive": 37}}))
    runs = collect_runs(tmp_path)
    summ = aggregate_metric(runs, "extinct")
    assert summ.n == 2
    assert summ.mean == 0.5


def test_aggregate_metric_extinct_explicit_wins(tmp_path):
    """Un champ 'extinct' explicite prime sur la dérivation."""
    _write_run(tmp_path / "s1", 1, alive=80, ext=True)  # incohérent exprès
    runs = collect_runs(tmp_path)
    summ = aggregate_metric(runs, "extinct")
    assert summ.n == 1
    assert summ.mean == 1.0


def test_aggregate_metric_missing_ignored(tmp_path):
    _write_run(tmp_path / "s1", 1, 80, shannon=1.0)
    _write_run(tmp_path / "s2", 2, 80)  # pas de shannon
    runs = collect_runs(tmp_path)
    summ = aggregate_metric(runs, "ecology_v25.shannon_diversity")
    assert summ.n == 1
