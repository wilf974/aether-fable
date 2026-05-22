"""Tests pour FoodGridConfig."""
from __future__ import annotations

import pytest

from aetherlife.config import FoodGridConfig


def test_default_config_instantiates() -> None:
    cfg = FoodGridConfig()
    assert cfg.rows == 16
    assert cfg.cols == 16
    assert cfg.max_energy == 100.0
    assert cfg.start_energy == 50.0
    assert cfg.metabolism == 1.0
    assert cfg.food_value == 20.0
    assert cfg.max_steps == 1000


def test_obs_dim() -> None:
    cfg = FoodGridConfig(rows=4, cols=5)
    assert cfg.obs_dim == 2 * 4 * 5 + 1


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("rows", 0),
        ("cols", -1),
        ("max_energy", 0),
        ("metabolism", 0),
        ("metabolism", -1.5),
        ("food_value", 0),
        ("food_value", -10),
        ("death_penalty", -1),
        ("initial_food_density", -0.1),
        ("initial_food_density", 1.5),
        ("food_respawn_lambda", -0.1),
        ("max_steps", 0),
    ],
)
def test_invalid_field_raises(field: str, bad_value: float) -> None:
    kwargs: dict = {field: bad_value}
    with pytest.raises(ValueError):
        FoodGridConfig(**kwargs)


def test_start_energy_must_be_positive_and_below_max() -> None:
    with pytest.raises(ValueError):
        FoodGridConfig(start_energy=0)
    with pytest.raises(ValueError):
        FoodGridConfig(start_energy=200, max_energy=100)


def test_start_position_must_be_in_grid() -> None:
    FoodGridConfig(rows=5, cols=5, start_position=(0, 0))
    FoodGridConfig(rows=5, cols=5, start_position=(4, 4))
    with pytest.raises(ValueError):
        FoodGridConfig(rows=5, cols=5, start_position=(5, 0))
    with pytest.raises(ValueError):
        FoodGridConfig(rows=5, cols=5, start_position=(-1, 0))


def test_config_is_frozen() -> None:
    cfg = FoodGridConfig()
    with pytest.raises(Exception):
        cfg.rows = 32
