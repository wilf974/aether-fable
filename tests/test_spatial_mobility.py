"""Tests du helper pur de mobilité spatiale (Historian, chantier A)."""
import pytest

from aetherlife.historian.spatial_mobility import (
    BINS,
    OccupancyAccumulator,
    build_spatial_mobility_block,
    is_village_basin,
    pearson_corr,
    window_bounds,
)


def test_window_bounds_is_thirds_excluding_founding():
    # Officiel : 1er tiers vs 3e tiers (le ~1er 10% de fondation est dilué,
    # on compare des phases installées — pas le transitoire de départ).
    assert window_bounds(16000) == ((0, 5333), (10667, 16000))
    assert window_bounds(60) == ((0, 20), (40, 60))


def test_pearson_corr_identical_is_one():
    assert pearson_corr([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_pearson_corr_anticorrelated_is_minus_one():
    assert pearson_corr([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_corr_zero_variance_returns_zero():
    # une distribution constante -> std=0 -> corr indéfinie, on renvoie 0.0
    assert pearson_corr([1, 1, 1], [1, 2, 3]) == 0.0


def test_is_village_basin_threshold_inclusive_at_0_8():
    assert is_village_basin(0.80) is True
    assert is_village_basin(0.95) is True
    assert is_village_basin(0.79) is False


def test_occupancy_accumulator_bins_positions():
    # grille 40x40, BINS=8 -> super-cellules de 5x5
    acc = OccupancyAccumulator(rows=40, cols=40)
    acc.add_positions([(0, 0), (0, 0), (39, 39)])
    assert acc.n == 3
    assert acc.counts[0] == 2  # (0,0) -> bin 0 (coin haut-gauche)
    assert acc.counts[BINS * BINS - 1] == 1  # (39,39) -> dernier bin


def test_build_block_structure_and_samples():
    start = OccupancyAccumulator(40, 40)
    start.add_positions([(0, 0), (0, 0)])
    end = OccupancyAccumulator(40, 40)
    end.add_positions([(39, 39), (39, 39)])
    block = build_spatial_mobility_block(
        start, end, start_window=(0, 1600), end_window=(14400, 16000)
    )
    assert block["start_window_ticks"] == [0, 1600]
    assert block["end_window_ticks"] == [14400, 16000]
    assert block["n_samples_start"] == 2
    assert block["n_samples_end"] == 2
    assert "corr_occupation_start_end" in block
    assert "village_basin" in block


def test_build_block_identical_occupation_is_village():
    # même zone occupée début et fin -> corr=1 -> village
    start = OccupancyAccumulator(40, 40)
    end = OccupancyAccumulator(40, 40)
    for acc in (start, end):
        acc.add_positions([(2, 2), (2, 3), (20, 20), (35, 35)])
    block = build_spatial_mobility_block(
        start, end, start_window=(0, 1600), end_window=(14400, 16000)
    )
    assert block["corr_occupation_start_end"] == 1.0
    assert block["village_basin"] is True


def test_build_block_empty_window_yields_none_corr():
    start = OccupancyAccumulator(40, 40)
    start.add_positions([(1, 1)])
    end = OccupancyAccumulator(40, 40)  # vide (extinction avant fenêtre fin)
    block = build_spatial_mobility_block(
        start, end, start_window=(0, 1600), end_window=(14400, 16000)
    )
    assert block["corr_occupation_start_end"] is None
    assert block["village_basin"] is None
    assert block["n_samples_end"] == 0
