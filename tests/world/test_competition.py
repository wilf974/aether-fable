"""Tests V8-B1.5 — Compétition locale (metabolism modulé densité)."""
from __future__ import annotations

import pytest

from aetherlife.world.competition import (
    CompetitionConfig, crowd_metabolism_factor,
)


def test_competition_config_defaults() -> None:
    cfg = CompetitionConfig()
    assert cfg.enabled is False
    assert cfg.radius >= 1
    assert cfg.metabolism_per_neighbor >= 0
    assert cfg.max_factor >= 1.0


def test_competition_config_validates() -> None:
    with pytest.raises(ValueError):
        CompetitionConfig(radius=-1)
    with pytest.raises(ValueError):
        CompetitionConfig(metabolism_per_neighbor=-0.1)
    with pytest.raises(ValueError):
        CompetitionConfig(max_factor=0.5)


def test_crowd_factor_disabled_is_one() -> None:
    cfg = CompetitionConfig(enabled=False)
    assert crowd_metabolism_factor(10, cfg) == 1.0


def test_crowd_factor_no_neighbors_is_one() -> None:
    cfg = CompetitionConfig(enabled=True)
    assert crowd_metabolism_factor(0, cfg) == 1.0


def test_crowd_factor_linear_in_neighbors() -> None:
    cfg = CompetitionConfig(
        enabled=True, metabolism_per_neighbor=0.05, max_factor=10.0,
    )
    assert crowd_metabolism_factor(2, cfg) == pytest.approx(1.10)
    assert crowd_metabolism_factor(5, cfg) == pytest.approx(1.25)


def test_crowd_factor_capped() -> None:
    cfg = CompetitionConfig(
        enabled=True, metabolism_per_neighbor=0.1, max_factor=1.5,
    )
    # 10 voisins → 2.0 brut, mais capped à 1.5
    assert crowd_metabolism_factor(10, cfg) == pytest.approx(1.5)
