"""Tests Historian — observer/reporter sans influence sur les agents."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from aetherlife.historian import (
    DiscoveriesDetector, Discovery, Historian,
)
from aetherlife.historian.discoveries import DiscoveryCategory


# ─── Fixtures ───────────────────────────────────────────────────────────


def _report_dialects() -> dict:
    """Stub d'un run avec dialectes émergents (concentration 99 %, L2=3.0)."""
    return {
        "config": {"n_ticks": 30000, "seed": 42, "device": "cpu",
                   "vision_radius": 4, "obs_dim": 424},
        "runtime": {"duration_s": 1900.0, "ticks_per_sec": 15.8},
        "criterion_1_inheritance": {
            "lifespan_by_birth_quartile": {
                "Q1_early": {"n": 100, "mean": 1379, "median": 601},
                "Q2": {"n": 100, "mean": 1334, "median": 923},
                "Q3": {"n": 100, "mean": 1679, "median": 953},
                "Q4_late": {"n": 100, "mean": 2395, "median": 1694},
            },
        },
        "criterion_2_divergence": {
            "final_kl_mean": 6.34,
            "divergence_curve_sample": [],
        },
        "criterion_3_selection": {
            "n_lineages_initial": 20,
            "n_lineages_final": 10,
            "dominant_lineage_pct": 97.1,
        },
        "criterion_4_memory": {"total_brain_steps": 1_899_284},
        "final_state": {
            "n_alive": 102,
            "n_births_total": 729,
            "n_deaths": 697,
            "top_lineages": [{"root_id": 42, "alive": 99, "pct": 97.1}],
            "affinity_distribution": {"0": 50, "1": 30, "2": 12, "3": 10},
            "n_affinities_alive": 4,
        },
        "language_metrics_v8b2": {
            "n_brains_with_vocab": 10,
            "total_vocalize_count": 698823,
            "tokens_per_1000_ticks": 23294,
            "vocalize_energy_cost_total": 34941,
            "mean_token_lineage_concentration": 0.9988,
            "mean_usage_entropy": 1.036,
            "max_possible_entropy": 1.386,
            "entropy_ratio": 0.7475,
            "mean_inter_lineage_distance": 3.11,
            "per_token_usage_top": {"0": 198168, "1": 158805, "2": 167248, "3": 174602},
        },
        "curves": {
            "alive": [[5000, 101], [15000, 102], [30000, 102]],
            "lineages": [[5000, 8], [15000, 7], [30000, 10]],
            "loss": [[5000, 0.06], [15000, 0.01], [30000, 0.004]],
            "divergence": [[5000, 2.0], [30000, 6.34]],
        },
    }


def _report_extinction() -> dict:
    """Stub run qui se termine en extinction terminale."""
    return {
        "config": {"n_ticks": 50000, "seed": 7, "device": "cpu"},
        "runtime": {"duration_s": 1500.0, "ticks_per_sec": 33},
        "criterion_3_selection": {
            "n_lineages_initial": 10, "n_lineages_final": 0,
            "dominant_lineage_pct": 0.0,
        },
        "final_state": {
            "n_alive": 0, "n_births_total": 1500, "n_deaths": 1700,
            "top_lineages": [],
        },
        "curves": {
            "alive": [[10000, 100], [40000, 80], [42000, 0]],
            "loss": [[40000, 5.0], [42000, 400]],
        },
    }


# ─── DiscoveriesDetector ────────────────────────────────────────────────


def test_detect_dialects_emerging() -> None:
    det = DiscoveriesDetector(_report_dialects())
    found = det.detect_language()
    slugs = [d.slug for d in found]
    assert "language_dialects_emerging" in slugs
    d = next(x for x in found if x.slug == "language_dialects_emerging")
    assert d.category == DiscoveryCategory.LANGUAGE
    assert d.confidence > 0.7
    assert "concentration" in d.headline.lower() or "dialect" in d.headline.lower()
    assert "concentration_per_lineage" in d.evidence


def test_detect_extinction_terminal() -> None:
    det = DiscoveriesDetector(_report_extinction())
    found = det.detect_extinction()
    assert any(d.slug == "extinction_terminal" for d in found)


def test_detect_dqn_divergence() -> None:
    det = DiscoveriesDetector(_report_extinction())
    found = det.detect_instability()
    assert any(d.slug == "instability_dqn_divergence" for d in found)


def test_detect_cognition_inheritance_positive() -> None:
    det = DiscoveriesDetector(_report_dialects())
    found = det.detect_cognition()
    assert any(d.slug == "cognition_inheritance_observable" for d in found)


def test_detect_regime_long_tail() -> None:
    det = DiscoveriesDetector(_report_dialects())
    found = det.detect_regime()
    # 10 lignées + 97.1 % dominance → queue longue
    assert any(d.slug == "regime_long_tail" for d in found)


def test_detect_selection_strong() -> None:
    det = DiscoveriesDetector(_report_dialects())
    found = det.detect_selection()
    # 20 → 10 lignées = 50 % réduction
    assert any(d.slug == "selection_strong" for d in found)


def test_detect_all_returns_list() -> None:
    det = DiscoveriesDetector(_report_dialects())
    found = det.detect_all()
    assert isinstance(found, list)
    assert len(found) >= 3


def test_failsafe_empty_report() -> None:
    """Un report vide ne doit pas crasher le détecteur."""
    det = DiscoveriesDetector({})
    found = det.detect_all()
    assert isinstance(found, list)


def test_discovery_validates_confidence() -> None:
    with pytest.raises(ValueError):
        Discovery(
            slug="x", category=DiscoveryCategory.LANGUAGE,
            confidence=1.5, headline="x",
        )


# ─── Historian ──────────────────────────────────────────────────────────


def test_historian_renders_all_files() -> None:
    h = Historian.from_report(_report_dialects(), run_id="test_run")
    with tempfile.TemporaryDirectory() as tmp:
        files = h.write_all(tmp)
        for name in (
            "summary.md", "scientific_report.md", "public_article.md",
            "discoveries.md", "lineages.md", "dialects.md",
            "metrics.json", "events.jsonl", "charts.csv",
        ):
            assert name in files, f"Missing file: {name}"
            assert os.path.exists(files[name])
            # Fichiers non vides
            assert os.path.getsize(files[name]) > 0


def test_summary_contains_run_id() -> None:
    h = Historian.from_report(_report_dialects(), run_id="testrun123")
    s = h.render_summary()
    assert "testrun123" in s


def test_scientific_report_contains_limits() -> None:
    h = Historian.from_report(_report_dialects(), run_id="r")
    s = h.render_scientific_report()
    assert "limites" in s.lower() or "mono-seed" in s.lower()
    assert "validation" in s.lower()


def test_public_article_uses_probabilistic_language() -> None:
    """Le public article doit utiliser 'pattern', 'hypothèse', etc."""
    h = Historian.from_report(_report_dialects(), run_id="r")
    s = h.render_public_article()
    # Pas d'affirmation absolue type "les agents ont découvert X"
    assert "découvert" not in s.lower() or "hypothèse" in s.lower()
    # Mention claire d'observation probabiliste
    assert "hypothèse" in s.lower() or "pattern" in s.lower() or "corrélation" in s.lower()


def test_dialects_md_present_when_language_active() -> None:
    h = Historian.from_report(_report_dialects(), run_id="r")
    s = h.render_dialects()
    assert "vocabulaire" in s.lower() or "token" in s.lower()
    assert "99" in s  # la concentration 99 % apparaît


def test_dialects_md_says_not_active_when_no_vocab() -> None:
    report_no_lang = _report_extinction()  # pas de language_metrics_v8b2
    h = Historian.from_report(report_no_lang, run_id="r")
    s = h.render_dialects()
    assert "n'était pas activé" in s or "language" in s.lower()


def test_metrics_json_structure() -> None:
    h = Historian.from_report(_report_dialects(), run_id="r")
    with tempfile.TemporaryDirectory() as tmp:
        h.write_all(tmp)
        with open(os.path.join(tmp, "metrics.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert "run_id" in data
        assert "protocol" in data
        assert "ecology" in data
        assert "language" in data
        assert "discoveries" in data
        assert isinstance(data["discoveries"], list)


def test_events_jsonl_sortable() -> None:
    h = Historian.from_report(_report_dialects(), run_id="r")
    with tempfile.TemporaryDirectory() as tmp:
        h.write_all(tmp)
        events = []
        with open(os.path.join(tmp, "events.jsonl"), encoding="utf-8") as f:
            for line in f:
                events.append(json.loads(line))
        assert len(events) > 0
        # Premier event a un type
        assert "type" in events[0]


def test_charts_csv_has_header() -> None:
    h = Historian.from_report(_report_dialects(), run_id="r")
    with tempfile.TemporaryDirectory() as tmp:
        h.write_all(tmp)
        with open(os.path.join(tmp, "charts.csv"), encoding="utf-8") as f:
            header = f.readline().strip()
        assert "tick" in header
        assert "alive" in header


def test_historian_failsafe_empty_report() -> None:
    """Un report vide doit produire des rapports 'non observable' sans crash."""
    h = Historian.from_report({}, run_id="empty")
    s = h.render_summary()
    assert "empty" in s
    assert "non observable" in s


# ─── Pas d'influence sur les agents ─────────────────────────────────────


def test_historian_does_not_import_agents() -> None:
    """Vérifie que historian n'importe pas aetherlife.agents (read-only sur JSON)."""
    import aetherlife.historian as h
    src = open(h.__file__, encoding="utf-8").read()
    # Pas d'import des agents dans __init__.py
    assert "from aetherlife.agents" not in src


def test_historian_input_immutable() -> None:
    """Modifier le report après init ne doit pas changer les discoveries."""
    report = _report_dialects()
    h = Historian.from_report(report, run_id="r")
    n_disc_before = len(h.discoveries)
    # Mutation externe
    report["language_metrics_v8b2"]["mean_token_lineage_concentration"] = 0.0
    n_disc_after = len(h.discoveries)
    assert n_disc_before == n_disc_after
