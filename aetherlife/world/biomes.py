"""V8-B1.5 — Biomes : configuration + worldgen Voronoi.

4 types de biomes avec stratégies optimales différentes :

| Biome  | metabolism | food λ | food value | movement | stratégie |
|--------|-----------:|-------:|-----------:|---------:|-----------|
| PLAIN  |       1.0  |    1.0 |        1.0 |      1.0 | équilibré |
| FOREST |       1.0  |    2.0 |        1.0 |      1.2 | forager   |
| DESERT |       1.3  |    0.3 |        1.0 |      1.0 | planteur  |
| TUNDRA |       1.5  |   0.15 |        1.5 |      1.0 | migrateur |

L'hypothèse scientifique : 4 stratégies différentes = 4 lignées peuvent
coexister dans 4 niches → fin du monopole cognitif observé en V8-B1.

Worldgen : Voronoi avec n_seed_points seeds aléatoires uniformément
distribuées. Chaque tile reçoit le biome de la seed la plus proche.
Résultat : zones contigües naturelles, frontières organiques.

Référence : spec
`docs/superpowers/specs/2026-05-23-aetherlife-v8-b1-5-ecological-niches-design.md`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class BiomeType(IntEnum):
    """Types de biomes (int8 dans la biome_map)."""
    PLAIN = 0
    FOREST = 1
    DESERT = 2
    TUNDRA = 3


@dataclass(frozen=True)
class BiomeParams:
    """Modificateurs locaux appliqués selon le biome du tile courant."""

    metabolism_factor: float = 1.0
    food_lambda_factor: float = 1.0
    food_value_factor: float = 1.0
    movement_cost: float = 1.0
    passable: bool = True

    def __post_init__(self) -> None:
        if self.metabolism_factor <= 0:
            raise ValueError(
                f"metabolism_factor doit être > 0 (got {self.metabolism_factor})"
            )
        if self.food_lambda_factor < 0:
            raise ValueError(
                f"food_lambda_factor doit être >= 0 (got {self.food_lambda_factor})"
            )
        if self.food_value_factor <= 0:
            raise ValueError(
                f"food_value_factor doit être > 0 (got {self.food_value_factor})"
            )
        if self.movement_cost <= 0:
            raise ValueError(
                f"movement_cost doit être > 0 (got {self.movement_cost})"
            )


@dataclass(frozen=True)
class BiomeConfig:
    """Configuration des biomes pour V8-B1.5."""

    enabled: bool = False
    n_seed_points: int = 8

    plain: BiomeParams = field(default_factory=lambda: BiomeParams())
    forest: BiomeParams = field(default_factory=lambda: BiomeParams(
        food_lambda_factor=2.0, movement_cost=1.2,
    ))
    desert: BiomeParams = field(default_factory=lambda: BiomeParams(
        food_lambda_factor=0.3, metabolism_factor=1.3,
    ))
    tundra: BiomeParams = field(default_factory=lambda: BiomeParams(
        food_lambda_factor=0.15, metabolism_factor=1.5,
        food_value_factor=1.5,
    ))

    def __post_init__(self) -> None:
        if self.n_seed_points <= 0:
            raise ValueError(
                f"n_seed_points doit être > 0 (got {self.n_seed_points})"
            )


# Singleton "neutre" pour le mode disabled
_NEUTRAL_PARAMS = BiomeParams()


def biome_params_for(biome_id: int, cfg: BiomeConfig) -> BiomeParams:
    """Retourne les BiomeParams correspondant à un biome_id.

    Si `cfg.enabled is False`, renvoie toujours les params neutres
    (équivalent PLAIN) pour préserver la compatibilité V8-B1.

    Lève `ValueError` si biome_id ∉ {0, 1, 2, 3}.
    """
    if not cfg.enabled:
        return _NEUTRAL_PARAMS
    if biome_id == int(BiomeType.PLAIN):
        return cfg.plain
    if biome_id == int(BiomeType.FOREST):
        return cfg.forest
    if biome_id == int(BiomeType.DESERT):
        return cfg.desert
    if biome_id == int(BiomeType.TUNDRA):
        return cfg.tundra
    raise ValueError(f"biome_id inconnu : {biome_id}")


def generate_biome_map(
    rows: int,
    cols: int,
    cfg: BiomeConfig,
    seed: int = 0,
) -> np.ndarray:
    """Worldgen Voronoi : N seeds aléatoires, chaque tile = biome du seed le plus proche.

    Args:
        rows, cols: dimensions de la carte
        cfg: BiomeConfig (n_seed_points utilisé)
        seed: pour reproductibilité (deux appels même seed = même carte)

    Returns:
        np.ndarray shape (rows, cols), dtype int8, valeurs ∈ {0, 1, 2, 3}
    """
    rng = np.random.default_rng(seed)
    n = cfg.n_seed_points
    # Tirer N coordonnées de seeds (float pour granularité fine)
    seed_rows = rng.uniform(0, rows, size=n)
    seed_cols = rng.uniform(0, cols, size=n)
    # Tirer un type de biome aléatoire par seed
    seed_biomes = rng.integers(0, 4, size=n).astype(np.int8)
    # Construire le biome_map par nearest-seed
    biome_map = np.zeros((rows, cols), dtype=np.int8)
    # Vectorisé : pour chaque tile, calculer distance² à tous les seeds
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    # rr.shape = (rows, cols), cc.shape = (rows, cols)
    # On calcule dist² broadcast : (rows, cols, n)
    dr = rr[..., None] - seed_rows[None, None, :]  # (rows, cols, n)
    dc = cc[..., None] - seed_cols[None, None, :]
    d2 = dr * dr + dc * dc
    nearest = np.argmin(d2, axis=-1)  # (rows, cols), index 0..n-1
    biome_map[:] = seed_biomes[nearest]
    return biome_map
