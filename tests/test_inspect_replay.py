"""Tests scripts/inspect_replay.py — inspecteur tick-par-tick events v8."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import pytest

import inspect_replay as ir


def _agent(aid, r, c, e=5.0, age=10, lin=0, aff=0):
    return {"id": aid, "lin": lin, "r": r, "c": c, "e": e,
            "er": round(e / 10, 3), "age": age, "aff": aff}


@pytest.fixture
def run_dir(tmp_path):
    """Run synthétique schema v2 : 4 snapshots, collapse à la fin, 1 vocal."""
    events = [
        {"t": 100, "season": 0, "n_alive": 4, "n_lin": 2,
         "agents": [_agent(0, 0, 0), _agent(1, 1, 1, aff=1),
                    _agent(2, 7, 7, lin=1), _agent(3, 6, 6, lin=1, aff=1)],
         "vocal": {"0": 2}, "spots": [{"r": 0, "c": 0, "n": 2}], "food": [0] * 64},
        {"t": 200, "season": 0, "n_alive": 4, "n_lin": 2,
         "agents": [_agent(0, 0, 1), _agent(1, 1, 2, aff=1),
                    _agent(2, 7, 6, lin=1), _agent(3, 6, 7, lin=1, aff=1)],
         "vocal": {"0": 2, "1": 3}, "spots": [], "food": [0] * 64},
        {"t": 300, "season": 1, "n_alive": 4, "n_lin": 2,
         "agents": [_agent(0, 0, 2), _agent(1, 1, 1, aff=1),
                    _agent(2, 7, 5, lin=1), _agent(3, 5, 7, lin=1, aff=1)],
         "vocal": {}, "spots": [], "food": [0] * 64},
        {"t": 400, "season": 1, "n_alive": 1, "n_lin": 1,
         "agents": [_agent(0, 0, 3)],
         "vocal": {}, "spots": [], "food": [0] * 64},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (tmp_path / "meta.json").write_text(json.dumps({
        "rows": 8, "cols": 8, "n_tokens": 4, "seed": 1,
        "regime": "coordination", "n_initial_affinities": 2, "schema_version": 2,
    }), encoding="utf-8")
    return tmp_path


def test_resolve_paths_dir(run_dir):
    ev, meta = ir.resolve_paths(run_dir)
    assert ev.name == "events.jsonl" and meta is not None


def test_resolve_paths_direct_file(run_dir):
    ev, meta = ir.resolve_paths(run_dir / "events.jsonl")
    assert ev.name == "events.jsonl" and meta is not None


def test_load_events_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ir.load_events(tmp_path / "nope.jsonl")


def test_summary(run_dir, capsys):
    assert ir.main([str(run_dir), "--summary"]) == 0
    out = capsys.readouterr().out
    assert "snapshots=4" in out
    assert "regime=coordination" in out
    assert "début=4" in out and "fin=1" in out


def test_default_is_summary(run_dir, capsys):
    assert ir.main([str(run_dir)]) == 0
    assert "RÉSUMÉ" in capsys.readouterr().out


def test_show_tick_nearest(run_dir, capsys):
    ir.main([str(run_dir), "--tick", "190"])
    out = capsys.readouterr().out
    assert "TICK 200" in out
    assert "id=0" in out


def test_trace_agent(run_dir, capsys):
    ir.main([str(run_dir), "--agent", "0"])
    out = capsys.readouterr().out
    assert out.count("t=") >= 4  # présent dans les 4 snapshots
    assert "vocal=2" in out


def test_trace_absent_agent(run_dir, capsys):
    ir.main([str(run_dir), "--agent", "999"])
    assert "absent" in capsys.readouterr().out


def test_find_extinction(run_dir, capsys):
    ir.main([str(run_dir), "--find", "extinction"])
    out = capsys.readouterr().out
    assert "chute brutale" in out and "t=400" in out


def test_find_vocal(run_dir, capsys):
    ir.main([str(run_dir), "--find", "vocal"])
    out = capsys.readouterr().out
    assert "vocalisations" in out


def test_ecology(run_dir, capsys):
    ir.main([str(run_dir), "--ecology"])
    out = capsys.readouterr().out
    assert "ÉCOLOGIE" in out
    assert "Shannon" in out
    assert "recouvrement niche" in out


def test_missing_dir_returns_1(tmp_path, capsys):
    assert ir.main([str(tmp_path / "ghost")]) == 1
