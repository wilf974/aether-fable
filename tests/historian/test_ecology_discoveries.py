"""Tests detect_ecology — détecteur V2.5 sur le bloc ecology_v25."""
from __future__ import annotations

from aetherlife.historian.discoveries import (
    DiscoveriesDetector,
    DiscoveryCategory,
)


def _eco_block(**overrides) -> dict:
    base = {
        "n_observations": 1000,
        "affinity_counts": [250, 250, 250, 250],
        "shannon_diversity": 1.386,
        "shannon_evenness": 1.0,
        "simpson_dominance": 0.25,
        "niche_overlap": {"0-1": 0.8, "0-2": 0.8, "1-2": 0.8},
        "mean_niche_overlap": 0.8,
        "mean_lineages": 10.0,
        "max_lineages": 16,
        "alive_bifurcation": {"changed": False, "index": -1, "score": 0.5,
                              "mean_before": 80, "mean_after": 80},
    }
    base.update(overrides)
    return base


def _detect(eco: dict):
    return DiscoveriesDetector({"ecology_v25": eco}).detect_ecology()


def test_no_block_no_discovery():
    assert DiscoveriesDetector({}).detect_ecology() == []


def test_balanced_high_overlap_no_discovery():
    assert _detect(_eco_block()) == []


def test_niche_partitioning_detected():
    eco = _eco_block(
        niche_overlap={"0-1": 0.1, "0-2": 0.2, "1-2": 0.15},
        mean_niche_overlap=0.15,
    )
    out = _detect(eco)
    assert len(out) == 1
    d = out[0]
    assert d.slug == "ecology_niche_partitioning"
    assert d.category == DiscoveryCategory.SPATIAL
    assert d.confidence > 0.8
    assert "PARTITION" in d.headline


def test_partitioning_needs_two_populated_affinities():
    eco = _eco_block(
        affinity_counts=[1000, 0, 0, 0],
        niche_overlap={},
        mean_niche_overlap=0.0,
    )
    assert _detect(eco) == []


def test_dominance_detected():
    eco = _eco_block(
        affinity_counts=[900, 50, 30, 20],
        simpson_dominance=0.82,
    )
    out = _detect(eco)
    slugs = [d.slug for d in out]
    assert "ecology_affinity_dominance" in slugs
    d = next(x for x in out if x.slug == "ecology_affinity_dominance")
    assert d.category == DiscoveryCategory.REGIME
    assert d.confidence == 0.82


def test_collapse_detected_as_extinction():
    eco = _eco_block(alive_bifurcation={
        "changed": True, "index": 10, "score": 4.5,
        "mean_before": 80.0, "mean_after": 8.0,
    })
    out = _detect(eco)
    d = next(x for x in out if x.slug == "ecology_population_shift")
    assert d.category == DiscoveryCategory.EXTINCTION
    assert "EFFONDREMENT" in d.headline
    assert d.confidence == 1.0  # score 4.5 -> clamp 1.0


def test_growth_detected_as_regime():
    eco = _eco_block(alive_bifurcation={
        "changed": True, "index": 5, "score": 2.4,
        "mean_before": 20.0, "mean_after": 70.0,
    })
    out = _detect(eco)
    d = next(x for x in out if x.slug == "ecology_population_shift")
    assert d.category == DiscoveryCategory.REGIME
    assert "ESSOR" in d.headline
    assert 0.5 < d.confidence < 0.7


def test_detect_all_includes_ecology():
    eco = _eco_block(
        niche_overlap={"0-1": 0.1},
        mean_niche_overlap=0.1,
    )
    out = DiscoveriesDetector({"ecology_v25": eco}).detect_all()
    assert any(d.slug == "ecology_niche_partitioning" for d in out)
