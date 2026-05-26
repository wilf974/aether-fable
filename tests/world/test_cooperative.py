"""Tests V8-C3 — Actions coopératives (CooperativeConfig + GatherSpot)."""
from __future__ import annotations

import pytest

from aetherlife.world.cooperative import CooperativeConfig, GatherSpot


def test_cooperative_config_defaults() -> None:
    cfg = CooperativeConfig()
    assert cfg.enabled is False
    assert cfg.min_partners_adjacent == 1
    assert cfg.signal_window_ticks == 5
    assert cfg.bonus_energy == 30.0
    assert cfg.spawn_lambda == 0.5
    assert cfg.decay_ticks == 50
    assert cfg.max_active_spots == 30


def test_cooperative_config_validates() -> None:
    with pytest.raises(ValueError):
        CooperativeConfig(min_partners_adjacent=0)
    with pytest.raises(ValueError):
        CooperativeConfig(signal_window_ticks=0)
    with pytest.raises(ValueError):
        CooperativeConfig(bonus_energy=0)
    with pytest.raises(ValueError):
        CooperativeConfig(spawn_lambda=-0.1)
    with pytest.raises(ValueError):
        CooperativeConfig(decay_ticks=0)
    with pytest.raises(ValueError):
        CooperativeConfig(max_active_spots=0)


def test_gather_spot_expires() -> None:
    spot = GatherSpot(pos=(5, 5), spawned_tick=100, expires_at=150)
    assert not spot.is_expired(120)
    assert not spot.is_expired(149)
    assert spot.is_expired(150)
    assert spot.is_expired(200)


def test_gather_spot_pos() -> None:
    spot = GatherSpot(pos=(3, 7), spawned_tick=0, expires_at=50)
    assert spot.pos == (3, 7)
