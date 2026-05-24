"""Tests V8-B1.5 — Biomes : config + worldgen + lookup."""
from __future__ import annotations

import numpy as np
import pytest

from aetherlife.world.biomes import (
    BiomeConfig, BiomeParams, BiomeType, generate_biome_map,
    biome_params_for,
)


# ─── BiomeParams / BiomeConfig validation ──────────────────────────────


def test_biome_params_defaults() -> None:
    p = BiomeParams()
    assert p.metabolism_factor == 1.0
    assert p.food_lambda_factor == 1.0
    assert p.food_value_factor == 1.0
    assert p.movement_cost == 1.0
    assert p.passable is True


def test_biome_params_validates() -> None:
    with pytest.raises(ValueError):
        BiomeParams(metabolism_factor=0)
    with pytest.raises(ValueError):
        BiomeParams(food_lambda_factor=-0.1)
    with pytest.raises(ValueError):
        BiomeParams(food_value_factor=0)
    with pytest.raises(ValueError):
        BiomeParams(movement_cost=0)


def test_biome_config_defaults() -> None:
    cfg = BiomeConfig()
    assert cfg.enabled is False
    assert cfg.n_seed_points == 8
    # V8-B1.6 : biomes équilibrés (aucun n'est intrinsèquement meilleur)
    # mais différents (stratégies distinctes requises)
    assert cfg.plain.metabolism_factor == 1.0
    assert cfg.forest.food_lambda_factor > 1.0    # FOREST = food abondante
    assert cfg.desert.food_lambda_factor < 1.0    # DESERT = food rare
    assert cfg.desert.food_value_factor > 1.0     # mais food riche (compense)
    assert cfg.tundra.food_value_factor > 1.0     # TUNDRA = très nutritif
    assert cfg.tundra.food_lambda_factor < cfg.forest.food_lambda_factor


def test_biome_config_validates() -> None:
    with pytest.raises(ValueError):
        BiomeConfig(n_seed_points=0)


# ─── Worldgen Voronoi ──────────────────────────────────────────────────


def test_generate_biome_map_shape() -> None:
    cfg = BiomeConfig(enabled=True)
    bmap = generate_biome_map(rows=20, cols=30, cfg=cfg, seed=0)
    assert bmap.shape == (20, 30)
    assert bmap.dtype == np.int8


def test_generate_biome_map_uses_4_types_max() -> None:
    cfg = BiomeConfig(enabled=True, n_seed_points=20)
    bmap = generate_biome_map(rows=40, cols=40, cfg=cfg, seed=42)
    unique = set(bmap.flatten().tolist())
    # Seuls les 4 types autorisés
    assert unique.issubset({0, 1, 2, 3})


def test_generate_biome_map_deterministic() -> None:
    cfg = BiomeConfig(enabled=True)
    a = generate_biome_map(rows=20, cols=20, cfg=cfg, seed=42)
    b = generate_biome_map(rows=20, cols=20, cfg=cfg, seed=42)
    assert np.array_equal(a, b)


def test_generate_biome_map_different_seeds_differ() -> None:
    cfg = BiomeConfig(enabled=True)
    a = generate_biome_map(rows=20, cols=20, cfg=cfg, seed=1)
    b = generate_biome_map(rows=20, cols=20, cfg=cfg, seed=999)
    assert not np.array_equal(a, b)


def test_generate_biome_map_has_diversity() -> None:
    """Avec n_seed_points=8, on devrait avoir au moins 2 biomes différents."""
    cfg = BiomeConfig(enabled=True, n_seed_points=8)
    bmap = generate_biome_map(rows=30, cols=30, cfg=cfg, seed=7)
    unique = set(bmap.flatten().tolist())
    assert len(unique) >= 2


# ─── biome_params_for lookup ───────────────────────────────────────────


def test_biome_params_for_each_type() -> None:
    cfg = BiomeConfig(enabled=True)
    p_plain = biome_params_for(int(BiomeType.PLAIN), cfg)
    p_forest = biome_params_for(int(BiomeType.FOREST), cfg)
    p_desert = biome_params_for(int(BiomeType.DESERT), cfg)
    p_tundra = biome_params_for(int(BiomeType.TUNDRA), cfg)
    assert p_plain is cfg.plain
    assert p_forest is cfg.forest
    assert p_desert is cfg.desert
    assert p_tundra is cfg.tundra


def test_biome_params_for_unknown_raises() -> None:
    cfg = BiomeConfig(enabled=True)
    with pytest.raises(ValueError):
        biome_params_for(99, cfg)


# ─── Disabled biomes : compat ──────────────────────────────────────────


def test_biome_disabled_means_all_plain() -> None:
    """Si biomes.enabled=False, on peut quand même appeler le lookup
    mais tous les paramètres sont neutres (= PLAIN)."""
    cfg = BiomeConfig(enabled=False)
    p = biome_params_for(int(BiomeType.FOREST), cfg)
    # Doit retourner les params PLAIN (neutre) au lieu de FOREST
    assert p.metabolism_factor == 1.0
    assert p.food_lambda_factor == 1.0
    assert p.movement_cost == 1.0
