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


def _report_cooperation_apprenable() -> dict:
    """Stub V8-C3 où la mécanique coop est apprise : succès >= 50 + clustering positif."""
    return {
        "config": {"n_ticks": 15000, "seed": 42, "device": "cpu"},
        "runtime": {"duration_s": 600.0, "ticks_per_sec": 25.0},
        "final_state": {
            "n_alive": 100, "n_births_total": 350, "n_deaths": 250,
            "top_lineages": [{"root_id": 1, "alive": 60, "pct": 60.0}],
        },
        "criterion_3_selection": {
            "n_lineages_initial": 10, "n_lineages_final": 5,
        },
        "cooperative_v8c3": {
            "enabled": True,
            "gather_successes_total": 120,
            "gather_failures_total": 80000,
            "gather_success_rate": 0.0015,
            "active_spots_final": 30,
            "bonus_energy": 80.0,
            "spawn_lambda": 1.0,
            "decay_ticks": 100,
        },
        "cooperative_metrics_v8c3": {
            "n_successes_observed": 120,
            "clustering_pre_success": {
                "n": 120, "mean_neighbors_r3": 3.5,
                "median_neighbors_r3": 4,
                "trend_q4_minus_q1": 1.2,  # > 0 → apprentissage
            },
            "vocalize_to_gather_delay": {
                "n_with_token": 30, "mean_min_delay": 4.0,
                "trend_q4_minus_q1": 0.5, "coverage": 0.25,
            },
            "token_entropy_pre_success": {
                "n_successes_with_token": 30, "n_tokens_counted": 60,
                "distribution": {"0": 0.3, "1": 0.25, "2": 0.25, "3": 0.2},
                "dominant_token": 0, "dominant_share": 0.3, "entropy": 1.35,
            },
            "success_chains": {
                "n_chains": 80, "max_chain_len": 2, "mean_chain_len": 1.5,
                "n_isolated_successes": 50, "n_cascade_successes": 0,
            },
        },
        "curves": {"alive": [], "loss": [], "lineages": [], "divergence": []},
    }


def _report_cooperation_protocol() -> dict:
    """Stub V8-C3 avec protocole émergent : token dominant + delay trend < 0."""
    rep = _report_cooperation_apprenable()
    rep["cooperative_metrics_v8c3"]["token_entropy_pre_success"] = {
        "n_successes_with_token": 80, "n_tokens_counted": 200,
        "distribution": {"0": 0.05, "1": 0.05, "2": 0.78, "3": 0.12},
        "dominant_token": 2, "dominant_share": 0.78, "entropy": 0.75,
    }
    rep["cooperative_metrics_v8c3"]["vocalize_to_gather_delay"] = {
        "n_with_token": 80, "mean_min_delay": 3.0,
        "trend_q4_minus_q1": -1.5, "coverage": 0.67,  # < 0 = apprentissage
    }
    return rep


def _report_cooperation_cascade() -> dict:
    """Stub V8-C3 avec attracteur cascade : > 20 % succès en chaînes ≥ 3."""
    rep = _report_cooperation_apprenable()
    rep["cooperative_metrics_v8c3"]["success_chains"] = {
        "n_chains": 30, "max_chain_len": 8, "mean_chain_len": 4.0,
        "n_isolated_successes": 10, "n_cascade_successes": 60,
        # cascade_ratio = 60 / 120 = 0.5 > 0.2
    }
    return rep


def _report_cooperation_no_pattern() -> dict:
    """Stub V8-C3 où la coop existe mais aucun pattern (faibles successes)."""
    rep = _report_cooperation_apprenable()
    rep["cooperative_v8c3"]["gather_successes_total"] = 5
    rep["cooperative_metrics_v8c3"]["n_successes_observed"] = 5
    rep["cooperative_metrics_v8c3"]["clustering_pre_success"]["n"] = 5
    rep["cooperative_metrics_v8c3"]["clustering_pre_success"]["trend_q4_minus_q1"] = 0.1
    return rep


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


# ─── V8-B2.2 causality detectors ────────────────────────────────────────


def _report_causality_strong() -> dict:
    """Stub avec signal causal fort (shift max 0.12, context 0.58)."""
    return {
        "config": {"n_ticks": 30000, "seed": 42, "device": "cpu"},
        "runtime": {"duration_s": 1900, "ticks_per_sec": 15},
        "criterion_3_selection": {
            "n_lineages_initial": 20, "n_lineages_final": 4,
            "dominant_lineage_pct": 95.0,
        },
        "final_state": {
            "n_alive": 100, "n_births_total": 700, "n_deaths": 600,
            "top_lineages": [{"root_id": 1, "alive": 95, "pct": 95.0}],
        },
        "language_metrics_v8b2": {
            "mean_token_lineage_concentration": 0.99,
            "mean_inter_lineage_distance": 3.0,
            "entropy_ratio": 0.75,
            "total_vocalize_count": 500000,
            "per_token_usage_top": {"0": 130000, "1": 120000, "2": 130000, "3": 120000},
        },
        "language_causality_v8b2_2": {
            "listener_shift_mean": 0.086,
            "listener_shift_max": 0.121,
            "listener_shift_per_token": {"0": 0.121, "1": 0.075, "2": 0.063, "3": 0.084},
            "context_consistency_mean": 0.47,
            "context_consistency_per_token": {"0": 0.48, "1": 0.45, "2": 0.49, "3": 0.46},
            "n_emissions_total": 1000000,
            "verdict": "intermediaire",
        },
        "curves": {"alive": [], "lineages": [], "loss": [], "divergence": []},
    }


def test_detect_causality_signal_present() -> None:
    det = DiscoveriesDetector(_report_causality_strong())
    found = det.detect_causality()
    slugs = [d.slug for d in found]
    assert "causality_signal_present" in slugs


def test_detect_causality_signal_strong() -> None:
    """shift_max > 0.10 → signal fort détecté."""
    det = DiscoveriesDetector(_report_causality_strong())
    found = det.detect_causality()
    slugs = [d.slug for d in found]
    assert "causality_signal_strong" in slugs
    d = next(x for x in found if x.slug == "causality_signal_strong")
    assert "0" in d.headline  # token 0 mentionné


def test_detect_causality_context_specialization() -> None:
    """context 0.47 sur 72 clusters = ×34 baseline → spécialisation détectée."""
    det = DiscoveriesDetector(_report_causality_strong())
    found = det.detect_causality()
    slugs = [d.slug for d in found]
    assert "causality_context_specialization" in slugs


def test_detect_causality_skip_if_no_emissions() -> None:
    report = _report_causality_strong()
    report["language_causality_v8b2_2"]["n_emissions_total"] = 100
    det = DiscoveriesDetector(report)
    found = det.detect_causality()
    # Pas assez d'émissions → aucun pattern détecté
    assert found == []


def test_detect_causality_decorative_signal_low() -> None:
    """shift très bas + context bas → aucun pattern détecté."""
    report = _report_causality_strong()
    report["language_causality_v8b2_2"]["listener_shift_mean"] = 0.001
    report["language_causality_v8b2_2"]["listener_shift_max"] = 0.002
    report["language_causality_v8b2_2"]["context_consistency_mean"] = 0.05
    det = DiscoveriesDetector(report)
    found = det.detect_causality()
    # Aucun pattern (shift < 0.03 ET context < 0.30, ratio < 10)
    assert found == []


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


# ─── V8-C3 — Détecteurs coopération ─────────────────────────────────────


def test_detect_cooperation_apprenable() -> None:
    """≥ 50 succès + clustering trend > 0 → mécanique apprise."""
    det = DiscoveriesDetector(_report_cooperation_apprenable())
    found = det.detect_cooperation()
    slugs = [d.slug for d in found]
    assert "cooperation_apprenable" in slugs
    d = next(x for x in found if x.slug == "cooperation_apprenable")
    assert d.category == DiscoveryCategory.COOPERATION
    assert 0.0 < d.confidence <= 1.0
    assert d.evidence["gather_successes_total"] == 120
    assert d.evidence["clustering_trend_q4_minus_q1"] == 1.2


def test_detect_cooperation_protocol_emergent() -> None:
    """Token dominant > 0.5 + delay trend < 0 → protocole émergent."""
    det = DiscoveriesDetector(_report_cooperation_protocol())
    found = det.detect_cooperation()
    slugs = [d.slug for d in found]
    assert "cooperation_protocol_emergent" in slugs
    d = next(x for x in found if x.slug == "cooperation_protocol_emergent")
    assert d.evidence["dominant_token"] == 2
    assert d.evidence["dominant_share_pre_success"] == 0.78
    assert d.evidence["delay_trend_q4_minus_q1"] < 0


def test_detect_cooperation_cascade_attractor() -> None:
    """cascade_ratio > 0.2 → attracteur détecté."""
    det = DiscoveriesDetector(_report_cooperation_cascade())
    found = det.detect_cooperation()
    slugs = [d.slug for d in found]
    assert "cooperation_cascade_attractor" in slugs
    d = next(x for x in found if x.slug == "cooperation_cascade_attractor")
    assert d.evidence["cascade_ratio"] == 0.5
    assert d.evidence["max_chain_len"] == 8


def test_detect_cooperation_mechanic_active_no_pattern() -> None:
    """Succès trop faibles → diagnostic neutre, pas surinterprétation."""
    det = DiscoveriesDetector(_report_cooperation_no_pattern())
    found = det.detect_cooperation()
    slugs = [d.slug for d in found]
    assert "cooperation_mechanic_active_no_pattern" in slugs
    # Ne doit JAMAIS déclencher les 3 patterns positifs avec si peu de succès
    assert "cooperation_apprenable" not in slugs
    assert "cooperation_protocol_emergent" not in slugs
    assert "cooperation_cascade_attractor" not in slugs


def test_detect_cooperation_skip_if_disabled() -> None:
    """Si cooperative.enabled=False, aucune découverte coop."""
    det = DiscoveriesDetector({"cooperative_v8c3": {"enabled": False}})
    found = det.detect_cooperation()
    assert found == []


def test_detect_cooperation_skip_if_absent() -> None:
    """Pas de section cooperative_v8c3 → silence absolu."""
    det = DiscoveriesDetector({})
    found = det.detect_cooperation()
    assert found == []


def test_detect_all_includes_cooperation_when_present() -> None:
    """detect_all() doit inclure les découvertes coop dans la sortie."""
    det = DiscoveriesDetector(_report_cooperation_protocol())
    found = det.detect_all()
    slugs = [d.slug for d in found]
    assert "cooperation_protocol_emergent" in slugs


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
