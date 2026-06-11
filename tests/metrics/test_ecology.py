"""Tests aetherlife.metrics.ecology — diversité, dominance, niche, bifurcation."""
from __future__ import annotations

import math

import pytest

from aetherlife.metrics.ecology import (
    EcologyTracker,
    detect_bifurcation,
    pianka_overlap,
    shannon_diversity,
    simpson_dominance,
)


# --- Shannon ---

def test_shannon_single_category_is_zero():
    assert shannon_diversity([10, 0, 0]) == 0.0


def test_shannon_uniform_equals_log_s():
    assert shannon_diversity([5, 5, 5, 5]) == pytest.approx(math.log(4))


def test_shannon_evenness_uniform_is_one():
    assert shannon_diversity([3, 3, 3], normalized=True) == pytest.approx(1.0)


def test_shannon_empty_is_zero():
    assert shannon_diversity([0, 0]) == 0.0
    assert shannon_diversity([]) == 0.0


def test_shannon_more_diverse_is_higher():
    assert shannon_diversity([8, 1, 1]) < shannon_diversity([4, 3, 3])


# --- Simpson ---

def test_simpson_monoculture_is_one():
    assert simpson_dominance([100, 0, 0]) == pytest.approx(1.0)


def test_simpson_uniform_is_inverse_s():
    assert simpson_dominance([5, 5, 5, 5]) == pytest.approx(0.25)


def test_simpson_empty_is_zero():
    assert simpson_dominance([]) == 0.0


# --- Pianka ---

def test_pianka_identical_is_one():
    assert pianka_overlap([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_pianka_scale_invariant():
    assert pianka_overlap([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_pianka_disjoint_is_zero():
    assert pianka_overlap([1, 0, 0], [0, 0, 1]) == pytest.approx(0.0)


def test_pianka_symmetric():
    a, b = [3, 1, 0], [1, 2, 4]
    assert pianka_overlap(a, b) == pytest.approx(pianka_overlap(b, a))


def test_pianka_empty_is_zero():
    assert pianka_overlap([0, 0], [1, 1]) == 0.0


def test_pianka_length_mismatch_raises():
    with pytest.raises(ValueError):
        pianka_overlap([1, 2], [1, 2, 3])


# --- Bifurcation ---

def test_bifurcation_detects_collapse():
    series = [80] * 10 + [5] * 10
    res = detect_bifurcation(series)
    assert res.changed
    assert res.index == 10
    assert res.mean_before > res.mean_after


def test_bifurcation_flat_series_no_change():
    res = detect_bifurcation([50] * 20)
    assert not res.changed
    assert res.index == -1


def test_bifurcation_too_short():
    res = detect_bifurcation([1, 2])
    assert not res.changed


def test_bifurcation_noisy_stable_below_threshold():
    series = [50, 51, 49, 50, 52, 48, 50, 51, 49, 50]
    res = detect_bifurcation(series, threshold=2.0)
    assert not res.changed


# --- EcologyTracker ---

def test_tracker_pure_niches_low_overlap():
    tr = EcologyTracker(rows=8, cols=8, n_affinities=2, grid_bins=4)
    # affinité 0 occupe le coin haut-gauche, affinité 1 le coin bas-droit
    for _ in range(20):
        tr.observe_agent(0, 0, 0)
        tr.observe_agent(7, 7, 1)
    block = tr.finalize()
    assert block["niche_overlap"]["0-1"] == pytest.approx(0.0)
    assert block["n_observations"] == 40
    assert block["affinity_counts"] == [20, 20]


def test_tracker_same_niche_high_overlap():
    tr = EcologyTracker(rows=8, cols=8, n_affinities=2, grid_bins=4)
    for _ in range(20):
        tr.observe_agent(0, 0, 0)
        tr.observe_agent(0, 1, 1)  # même super-cellule (bin 0)
    block = tr.finalize()
    assert block["niche_overlap"]["0-1"] == pytest.approx(1.0)


def test_tracker_consumes_v8_event():
    tr = EcologyTracker(rows=8, cols=8, n_affinities=4, grid_bins=4)
    ev = {
        "t": 5, "n_alive": 3, "n_lin": 2,
        "agents": [
            {"id": 0, "lin": 0, "r": 0, "c": 0, "e": 5.0, "er": 0.5, "age": 3, "aff": 0},
            {"id": 1, "lin": 0, "r": 1, "c": 1, "e": 4.0, "er": 0.4, "age": 3, "aff": 0},
            {"id": 2, "lin": 1, "r": 7, "c": 7, "e": 6.0, "er": 0.6, "age": 3, "aff": 1},
        ],
    }
    tr.observe_event(ev)
    block = tr.finalize()
    assert block["n_observations"] == 3
    assert block["affinity_counts"][0] == 2
    assert block["affinity_counts"][1] == 1
    assert block["max_lineages"] == 2


def test_tracker_event_handles_null_affinity():
    tr = EcologyTracker(rows=8, cols=8, n_affinities=4)
    tr.observe_event({"t": 1, "n_alive": 1, "agents": [
        {"id": 0, "lin": 0, "r": 2, "c": 2, "e": 1.0, "er": 0.1, "age": 1, "aff": None},
    ]})
    block = tr.finalize()
    assert block["affinity_counts"][0] == 1


def test_tracker_finalize_keys_stable():
    tr = EcologyTracker(rows=4, cols=4)
    block = tr.finalize()
    expected = {
        "n_observations", "affinity_counts", "shannon_diversity",
        "shannon_evenness", "simpson_dominance", "niche_overlap",
        "mean_niche_overlap", "mean_lineages", "max_lineages",
        "alive_bifurcation",
    }
    assert expected <= set(block)
