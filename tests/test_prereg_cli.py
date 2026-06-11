"""Tests scripts/prereg.py — CLI plan / lock / audit."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import pytest

from aetherlife.analysis.prereg import Comparator, Condition, Criterion, PreregSpec
import prereg as cli


@pytest.fixture
def spec_file(tmp_path):
    spec = PreregSpec(
        prereg_id="demo",
        hypothesis="k4 survit mieux que k1",
        primary_metric="extinct",
        conditions=(Condition("k1", {"n_initial_affinities": 1}),
                    Condition("k4", {"n_initial_affinities": 4})),
        criteria=(Criterion("k4 survit", "extinct", Comparator.LT, 0.2, "k4"),),
        seeds=(1, 2),
    )
    p = tmp_path / "spec.json"
    spec.save(p)
    return p


def _write_run(d, extinct):
    d.mkdir(parents=True, exist_ok=True)
    (d / "overnight_v8b1_seed1.json").write_text(json.dumps({"extinct": extinct}))


def test_plan_prints_commands(spec_file, capsys):
    assert cli.main(["plan", str(spec_file)]) == 0
    out = capsys.readouterr().out
    assert out.count("overnight_v8b1.py") == 4  # 2 cond × 2 seeds
    assert "NON VERROUILLÉ" in out


def test_lock_then_plan_shows_locked(spec_file, capsys):
    assert cli.main(["lock", str(spec_file)]) == 0
    assert "Verrouillé" in capsys.readouterr().out
    cli.main(["plan", str(spec_file)])
    assert "VERROUILLÉ :" in capsys.readouterr().out


def test_lock_idempotent(spec_file, capsys):
    cli.main(["lock", str(spec_file)])
    capsys.readouterr()
    cli.main(["lock", str(spec_file)])
    assert "Déjà verrouillé" in capsys.readouterr().err


def test_audit_success(spec_file, tmp_path, capsys):
    runs = tmp_path / "runs"
    _write_run(runs / "k4" / "seed1", False)
    _write_run(runs / "k4" / "seed2", False)
    _write_run(runs / "k1" / "seed1", True)
    assert cli.main(["audit", str(spec_file), "--runs", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "VERDICT : SUCCÈS" in out


def test_audit_missing_runs_dir(spec_file, tmp_path, capsys):
    assert cli.main(["audit", str(spec_file), "--runs", str(tmp_path / "ghost")]) == 1


def test_spec_not_found(tmp_path, capsys):
    assert cli.main(["plan", str(tmp_path / "none.json")]) == 1
