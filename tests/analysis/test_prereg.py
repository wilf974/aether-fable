"""Tests aetherlife.analysis.prereg — contrat auditable."""
from __future__ import annotations

import pytest

from aetherlife.analysis.prereg import (
    Comparator, Condition, Criterion, PreregSpec, Verdict, audit,
)


def _spec(**kw) -> PreregSpec:
    base = dict(
        prereg_id="c2_repli_N30",
        hypothesis="diversité d'affinité -> survie",
        primary_metric="extinct",
        conditions=(Condition("k1", {"n_initial_affinities": 1}),
                    Condition("k4", {"n_initial_affinities": 4})),
        criteria=(
            Criterion("k4 survit", "extinct", Comparator.LT, 0.2, "k4"),
            Criterion("k1 risqué", "extinct", Comparator.GT, 0.4, "k1"),
        ),
        seeds=(1, 2, 3),
    )
    base.update(kw)
    return PreregSpec(**base)


# ── validation ──
def test_spec_requires_conditions():
    with pytest.raises(ValueError):
        _spec(conditions=())


def test_spec_unique_labels():
    with pytest.raises(ValueError):
        _spec(conditions=(Condition("k1"), Condition("k1")))


def test_criterion_unknown_condition_rejected():
    with pytest.raises(ValueError):
        _spec(criteria=(Criterion("x", "extinct", Comparator.LT, 0.2, "kX"),))


def test_comparator_logic():
    assert Comparator.GT.test(0.5, 0.4)
    assert not Comparator.LT.test(0.5, 0.4)
    assert Comparator.GE.test(0.4, 0.4)
    assert Comparator.LE.test(0.4, 0.4)


# ── sérialisation ──
def test_roundtrip_json(tmp_path):
    spec = _spec()
    p = spec.save(tmp_path / "spec.json")
    reloaded = PreregSpec.load(p)
    assert reloaded.to_dict() == spec.to_dict()
    assert reloaded.conditions[0].overrides == {"n_initial_affinities": 1}
    assert reloaded.criteria[0].comparator is Comparator.LT


def test_locked_sets_timestamp():
    spec = _spec()
    assert spec.locked_at is None
    locked = spec.locked(when="2026-06-11T00:00:00")
    assert locked.locked_at == "2026-06-11T00:00:00"
    assert spec.locked_at is None  # immuable : l'original n'est pas modifié


# ── plan de lancement ──
def test_launch_plan_cartesian():
    spec = _spec()
    plan = spec.launch_plan(out_root="results")
    assert len(plan) == 2 * 3  # 2 conditions × 3 seeds
    first = plan[0]
    assert first["label"] == "k1" and first["seed"] == 1
    assert "results/c2_repli_N30/k1/seed1" == first["out_dir"]
    assert "--n-initial-affinities" in first["command"]
    assert "1" in first["command"]


# ── audit ──
def _runs(*extinct_flags):
    return [{"extinct": e} for e in extinct_flags]


def test_audit_success():
    spec = _spec()
    by_cond = {
        "k1": _runs(True, True, False),   # extinction 0.667 > 0.4 ✓
        "k4": _runs(False, False, False), # extinction 0.0 < 0.2 ✓
    }
    res = audit(spec, by_cond)
    assert res.verdict is Verdict.SUCCESS
    assert res.n_pass == 2 and res.n_total == 2


def test_audit_partial():
    spec = _spec()
    by_cond = {
        "k1": _runs(True, True, False),   # 0.667 > 0.4 ✓
        "k4": _runs(True, True, False),   # 0.667 < 0.2 ✗
    }
    res = audit(spec, by_cond)
    assert res.verdict is Verdict.PARTIAL
    assert res.n_pass == 1


def test_audit_failure():
    spec = _spec()
    by_cond = {
        "k1": _runs(False, False, False),  # 0.0 > 0.4 ✗
        "k4": _runs(True, True, True),     # 1.0 < 0.2 ✗
    }
    res = audit(spec, by_cond)
    assert res.verdict is Verdict.FAILURE
    assert res.n_pass == 0


def test_audit_incomplete_when_missing():
    spec = _spec()
    by_cond = {"k1": _runs(True, True), "k4": []}  # k4 sans données
    res = audit(spec, by_cond)
    assert res.verdict is Verdict.INCOMPLETE


def test_audit_to_dict_serializable():
    import json
    spec = _spec()
    res = audit(spec, {"k1": _runs(True), "k4": _runs(False)})
    json.dumps(res.to_dict())  # ne lève pas
    assert res.to_dict()["verdict"] in ("SUCCÈS", "PARTIEL", "ÉCHEC", "INCOMPLET")
