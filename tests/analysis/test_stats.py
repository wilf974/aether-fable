"""Tests aetherlife.analysis.stats."""
from __future__ import annotations

import pytest

from aetherlife.analysis.stats import (
    bootstrap_ci, proportion, summarize, wilson_ci,
)


# ── Wilson ──
def test_wilson_basic():
    pr = wilson_ci(8, 10)
    assert pr.p == 0.8
    assert pr.lo < 0.8 < pr.hi
    assert 0.0 <= pr.lo and pr.hi <= 1.0


def test_wilson_zero_n():
    pr = wilson_ci(0, 0)
    assert pr.p == 0.0 and pr.lo == 0.0 and pr.hi == 0.0


def test_wilson_all_success_bounded():
    pr = wilson_ci(10, 10)
    assert pr.p == 1.0 and pr.hi <= 1.0 and pr.lo < 1.0


def test_wilson_rejects_bad_counts():
    with pytest.raises(ValueError):
        wilson_ci(5, 3)


def test_wilson_narrower_with_more_data():
    narrow = wilson_ci(80, 100)
    wide = wilson_ci(8, 10)
    assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)


# ── summarize ──
def test_summarize_empty():
    s = summarize([])
    assert s.n == 0 and s.method == "degenerate"


def test_summarize_single():
    s = summarize([5.0])
    assert s.n == 1 and s.mean == 5.0 and s.ci_lo == 5.0 and s.ci_hi == 5.0


def test_summarize_mean_std():
    s = summarize([2, 4, 4, 4, 5, 5, 7, 9], method="normal")
    assert s.mean == 5.0
    assert round(s.std, 4) == 2.1381  # ddof=1
    assert s.ci_lo < 5.0 < s.ci_hi


def test_summarize_bootstrap_deterministic():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    a = summarize(data, seed=42)
    b = summarize(data, seed=42)
    assert (a.ci_lo, a.ci_hi) == (b.ci_lo, b.ci_hi)
    assert a.ci_lo < a.mean < a.ci_hi


def test_summarize_normal_ci_symmetric():
    s = summarize([10, 20, 30, 40, 50], method="normal")
    assert round(s.mean - s.ci_lo, 6) == round(s.ci_hi - s.mean, 6)


# ── proportion ──
def test_proportion_default_bool():
    pr = proportion([True, True, False, False, True])
    assert pr.successes == 3 and pr.n == 5 and pr.p == 0.6


def test_proportion_predicate():
    pr = proportion([1, 2, 3, 4, 5], predicate=lambda x: x > 3)
    assert pr.successes == 2 and pr.n == 5


def test_bootstrap_ci_constant():
    lo, hi = bootstrap_ci([7, 7, 7, 7])
    assert lo == 7.0 and hi == 7.0
