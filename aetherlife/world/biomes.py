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
    WATER = 4   # V8-B1.6 — infranchissable (passable=False)


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

    # V8-B1.6 : biomes calibrés pour que metabolism × (1/food_lambda) soit
    # ~ constant. Aucun biome n'est intrinsèquement meilleur qu'un autre :
    # ils nécessitent des stratégies différentes mais sont équivalents
    # en somme (en pression écologique).
    plain: BiomeParams = field(default_factory=lambda: BiomeParams(
        food_lambda_factor=1.0, metabolism_factor=1.0, food_value_factor=1.0,
    ))
    forest: BiomeParams = field(default_factory=lambda: BiomeParams(
        food_lambda_factor=1.5, metabolism_factor=1.1, food_value_factor=0.9,
        movement_cost=1.2,
    ))
    desert: BiomeParams = field(default_factory=lambda: BiomeParams(
        food_lambda_factor=0.6, metabolism_factor=0.9, food_value_factor=1.2,
    ))
    tundra: BiomeParams = field(default_factory=lambda: BiomeParams(
        food_lambda_factor=0.5, metabolism_factor=1.1, food_value_factor=1.4,
    ))
    water: BiomeParams = field(default_factory=lambda: BiomeParams(
        passable=False, food_lambda_factor=0.0,
    ))

    # V8-B1.6 — Worldgen options (kept for future B1.7)
    worldgen: str = "voronoi"   # ou "continental"
    passage_width: int = 2

    # V8-B1.6 — Spéciation par affinity héritée
    # Chaque lignée naît avec un biome_affinity ∈ {0..3}. Bonus dans son
    # biome, malus hors. Reproduction permise SEULEMENT dans son biome.
    affinity_enabled: bool = False
    in_affinity_metabolism: float = 0.7      # -30 % dans son biome
    in_affinity_food_value: float = 1.3      # +30 % dans son biome
    out_affinity_metabolism: float = 1.5     # +50 % hors biome
    out_affinity_food_value: float = 0.7     # -30 % hors biome
    out_affinity_movement_mult: float = 2.5  # ×2.5 le movement cost hors
    reproduction_locked_to_affinity: bool = True

    # V8-B1.6 — Worldgen équilibré (force ≥1 seed de chaque type)
    balanced_seeds: bool = False

    # V8-C2 — Food invisible (seul listen peut la révéler).
    # Si True, le canal "food" de l'observation égocentrique est masqué
    # sauf pour la tile centrale (sous l'agent). L'agent ne peut donc
    # PAS voir la food à distance, seulement la consommer en marchant
    # dessus OU être guidé par les vocalises des voisins.
    hidden_food: bool = False

    # V8-B1.7 — Seed bank + respawn affinities éteintes
    respawn_enabled: bool = False
    respawn_check_every: int = 100        # check tous les N ticks
    respawn_extinct_after_ticks: int = 5000  # seuil avant respawn
    respawn_threshold: int = 2            # respawn si n_alive_aff < threshold
    respawn_initial_energy: float = 200.0
    seed_bank_max_per_affinity: int = 2
    seed_bank_mutation_std: float = 0.05  # mutation du brain réveillé

    def __post_init__(self) -> None:
        if self.n_seed_points <= 0:
            raise ValueError(
                f"n_seed_points doit être > 0 (got {self.n_seed_points})"
            )
        if self.worldgen not in ("voronoi", "continental"):
            raise ValueError(
                f"worldgen doit être 'voronoi' ou 'continental' "
                f"(got '{self.worldgen}')"
            )
        if self.passage_width < 1:
            raise ValueError(
                f"passage_width doit être >= 1 (got {self.passage_width})"
            )
        if self.in_affinity_metabolism <= 0:
            raise ValueError(
                f"in_affinity_metabolism doit être > 0 "
                f"(got {self.in_affinity_metabolism})"
            )
        if self.in_affinity_food_value <= 0:
            raise ValueError(
                f"in_affinity_food_value doit être > 0 "
                f"(got {self.in_affinity_food_value})"
            )
        if self.out_affinity_metabolism <= 0:
            raise ValueError(
                f"out_affinity_metabolism doit être > 0 "
                f"(got {self.out_affinity_metabolism})"
            )
        if self.out_affinity_food_value <= 0:
            raise ValueError(
                f"out_affinity_food_value doit être > 0 "
                f"(got {self.out_affinity_food_value})"
            )
        if self.out_affinity_movement_mult < 1.0:
            raise ValueError(
                f"out_affinity_movement_mult doit être >= 1.0 "
                f"(got {self.out_affinity_movement_mult})"
            )


# Singleton "neutre" pour le mode disabled
_NEUTRAL_PARAMS = BiomeParams()


def biome_params_for(biome_id: int, cfg: BiomeConfig) -> BiomeParams:
    """Retourne les BiomeParams correspondant à un biome_id.

    Si `cfg.enabled is False`, renvoie toujours les params neutres
    (équivalent PLAIN) pour préserver la compatibilité V8-B1.

    Lève `ValueError` si biome_id ∉ {0..4} (WATER inclus en V8-B1.6).
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
    if biome_id == int(BiomeType.WATER):
        return cfg.water
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
        cfg: BiomeConfig (n_seed_points + balanced_seeds + worldgen)
        seed: pour reproductibilité (deux appels même seed = même carte)

    Returns:
        np.ndarray shape (rows, cols), dtype int8, valeurs ∈ {0..3} (PLAIN..TUNDRA).
        En V8-B1.6, WATER n'est pas généré par Voronoi (réservé continental).

    V8-B1.6 : si cfg.balanced_seeds=True, force ≥1 seed de chaque type
    (PLAIN, FOREST, DESERT, TUNDRA) parmi les n_seed_points.
    """
    rng = np.random.default_rng(seed)
    n = cfg.n_seed_points
    # Tirer N coordonnées de seeds (float pour granularité fine)
    seed_rows = rng.uniform(0, rows, size=n)
    seed_cols = rng.uniform(0, cols, size=n)
    # V8-B1.6 — Round-robin si balanced_seeds + complément random
    if cfg.balanced_seeds:
        base = np.array([0, 1, 2, 3], dtype=np.int8)
        if n <= 4:
            seed_biomes = base[:n].copy()
        else:
            extra = rng.integers(0, 4, size=n - 4).astype(np.int8)
            seed_biomes = np.concatenate([base, extra])
        rng.shuffle(seed_biomes)
    else:
        seed_biomes = rng.integers(0, 4, size=n).astype(np.int8)
    # Construire le biome_map par nearest-seed
    biome_map = np.zeros((rows, cols), dtype=np.int8)
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dr = rr[..., None] - seed_rows[None, None, :]
    dc = cc[..., None] - seed_cols[None, None, :]
    d2 = dr * dr + dc * dc
    nearest = np.argmin(d2, axis=-1)
    biome_map[:] = seed_biomes[nearest]
    return biome_map
