"""Tests du détecteur de mobilité spatiale (Historian, chantier A)."""
import pytest

from aetherlife.historian.discoveries import DiscoveriesDetector, DiscoveryCategory


def _report(corr, village, n_start=160, n_end=160):
    return {
        "spatial_mobility_v8c3": {
            "corr_occupation_start_end": corr,
            "village_basin": village,
            "start_window_ticks": [0, 1600],
            "end_window_ticks": [14400, 16000],
            "n_samples_start": n_start,
            "n_samples_end": n_end,
        }
    }


def test_mobility_village_emits_low_confidence():
    discs = DiscoveriesDetector(_report(0.95, True)).detect_mobility()
    assert len(discs) == 1
    d = discs[0]
    assert d.slug == "coordination_mobility"
    assert d.category == DiscoveryCategory.SPATIAL
    # confidence = 1 - corr -> faible (peu de mobilité)
    assert d.confidence == pytest.approx(0.05, abs=0.01)
    assert d.evidence["mobility_score"] == 0.95
    assert d.evidence["village_basin"] is True


def test_mobility_migration_emits_high_confidence():
    discs = DiscoveriesDetector(_report(0.30, False)).detect_mobility()
    d = discs[0]
    assert d.confidence == pytest.approx(0.70, abs=0.01)
    assert "mobil" in d.headline.lower()


def test_mobility_village_headline_mentions_stability():
    d = DiscoveriesDetector(_report(0.95, True)).detect_mobility()[0]
    assert "village" in d.headline.lower() or "stable" in d.headline.lower()


def test_mobility_negative_corr_confidence_clamped_to_one():
    d = DiscoveriesDetector(_report(-0.2, False)).detect_mobility()[0]
    assert d.confidence == 1.0


def test_mobility_absent_block_no_discovery():
    assert DiscoveriesDetector({}).detect_mobility() == []


def test_mobility_none_corr_no_discovery():
    # fenêtre vide (extinction) -> corr None -> pas de discovery
    assert DiscoveriesDetector(_report(None, None, n_end=0)).detect_mobility() == []


def test_mobility_included_in_detect_all():
    discs = DiscoveriesDetector(_report(0.30, False)).detect_all()
    assert any(x.slug == "coordination_mobility" for x in discs)
