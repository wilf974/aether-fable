"""Tests V8-B1.6 — Spéciation par affinity héritée."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aetherlife.world.biomes import (
    BiomeConfig, BiomeType, generate_biome_map,
)
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig, SeasonalMultiAgentConfig, SeasonalMultiAgentFoodGrid,
)


# ─── BiomeConfig affinity params validation ─────────────────────────────


def test_biome_config_affinity_defaults() -> None:
    cfg = BiomeConfig()
    assert cfg.affinity_enabled is False
    assert cfg.in_affinity_metabolism < 1.0
    assert cfg.in_affinity_food_value > 1.0
    assert cfg.out_affinity_metabolism > 1.0
    assert cfg.out_affinity_food_value < 1.0
    assert cfg.out_affinity_movement_mult >= 1.0
    assert cfg.reproduction_locked_to_affinity is True


def test_biome_config_affinity_validates() -> None:
    with pytest.raises(ValueError):
        BiomeConfig(in_affinity_metabolism=0)
    with pytest.raises(ValueError):
        BiomeConfig(in_affinity_food_value=-0.1)
    with pytest.raises(ValueError):
        BiomeConfig(out_affinity_metabolism=0)
    with pytest.raises(ValueError):
        BiomeConfig(out_affinity_food_value=-0.1)
    with pytest.raises(ValueError):
        BiomeConfig(out_affinity_movement_mult=0.5)


# ─── balanced_seeds : worldgen avec ≥1 seed de chaque type ──────────────


def test_balanced_seeds_guarantees_all_4_types() -> None:
    cfg = BiomeConfig(enabled=True, n_seed_points=4, balanced_seeds=True)
    bmap = generate_biome_map(rows=40, cols=40, cfg=cfg, seed=1)
    unique = set(bmap.flatten().tolist())
    assert unique == {0, 1, 2, 3}, f"Manque biome(s) : {unique}"


def test_balanced_seeds_with_more_seeds() -> None:
    cfg = BiomeConfig(enabled=True, n_seed_points=8, balanced_seeds=True)
    bmap = generate_biome_map(rows=40, cols=40, cfg=cfg, seed=2)
    unique = set(bmap.flatten().tolist())
    assert {0, 1, 2, 3}.issubset(unique)


# ─── Affinity : init à reset() ──────────────────────────────────────────


def _make_speciation_env(seed: int = 0) -> SeasonalMultiAgentFoodGrid:
    cfg = SeasonalMultiAgentConfig(
        rows=20, cols=20, n_agents=12, max_energy=300.0, start_energy=160.0,
        metabolism=0.4, food_value=18.0, death_penalty=0.0,
        initial_food_density=0.1, food_respawn_lambda=0.4, max_steps=2000,
        seasonal=SeasonalConfig(season_period=100),
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=130.0, energy_cost=60.0,
            cooldown_ticks=20, max_population=30,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=20.0,
            rest_bonus=3.0, cooldown_ticks=10,
        ),
        cache=CacheConfig(enabled=False),
        planting=PlantingConfig(enabled=False),
        biomes=BiomeConfig(
            enabled=True, n_seed_points=8, balanced_seeds=True,
            affinity_enabled=True,
            reproduction_locked_to_affinity=True,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    return env


def test_affinity_init_at_reset() -> None:
    env = _make_speciation_env()
    for a in env._agents:  # noqa: SLF001
        assert a.biome_affinity is not None
        assert 0 <= a.biome_affinity < 4


def test_affinity_disabled_means_none() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=10, cols=10, n_agents=4,
        biomes=BiomeConfig(enabled=True, affinity_enabled=False),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for a in env._agents:  # noqa: SLF001
        assert a.biome_affinity is None


def test_affinity_inherited_at_reproduction() -> None:
    env = _make_speciation_env(seed=0)
    parent = env._agents[0]  # noqa: SLF001
    # Forcer le parent sur SON biome pour permettre la repro
    found = False
    for r in range(env.cfg.rows):
        for c in range(env.cfg.cols):
            if int(env._biome_map[r, c]) == parent.biome_affinity:  # noqa: SLF001
                parent.pos = (r, c)
                found = True
                break
        if found:
            break
    assert found
    parent.energy = 200.0
    parent.last_repro_tick = -10**9
    env._step_count = 1  # noqa: SLF001
    env._try_reproductions()  # noqa: SLF001
    children = [a for a in env._agents if a.parent_id == 0]  # noqa: SLF001
    assert len(children) >= 1
    for child in children:
        assert child.biome_affinity == parent.biome_affinity


def test_reproduction_locked_outside_affinity() -> None:
    """Si parent est hors son biome, repro échoue."""
    env = _make_speciation_env(seed=0)
    parent = env._agents[0]  # noqa: SLF001
    # Forcer le parent sur un biome QUI N'EST PAS son affinity
    wrong_pos = None
    for r in range(env.cfg.rows):
        for c in range(env.cfg.cols):
            if int(env._biome_map[r, c]) != parent.biome_affinity:  # noqa: SLF001
                wrong_pos = (r, c)
                break
        if wrong_pos:
            break
    assert wrong_pos is not None
    parent.pos = wrong_pos
    parent.energy = 250.0
    parent.last_repro_tick = -10**9
    children_before = sum(
        1 for a in env._agents if a.parent_id == 0  # noqa: SLF001
    )
    env._step_count = 1  # noqa: SLF001
    env._try_reproductions()  # noqa: SLF001
    children_after = sum(
        1 for a in env._agents if a.parent_id == 0  # noqa: SLF001
    )
    # Aucune nouvelle naissance car parent hors affinity
    assert children_after == children_before


# ─── Bonus/malus metabolism ──────────────────────────────────────────────


def test_in_affinity_bonus_lower_metabolism() -> None:
    """Un agent dans son biome perd moins d'énergie qu'un agent hors biome."""
    env = _make_speciation_env(seed=0)
    a_in = env._agents[0]  # noqa: SLF001
    a_out = env._agents[1]  # noqa: SLF001
    # Force affinities différentes
    a_in.biome_affinity = 0  # PLAIN
    a_out.biome_affinity = 0
    # Trouve un PLAIN tile et un non-PLAIN tile
    plain_pos = None
    other_pos = None
    for r in range(env.cfg.rows):
        for c in range(env.cfg.cols):
            if int(env._biome_map[r, c]) == 0 and plain_pos is None:  # noqa: SLF001
                plain_pos = (r, c)
            if int(env._biome_map[r, c]) != 0 and other_pos is None:  # noqa: SLF001
                other_pos = (r, c)
            if plain_pos and other_pos:
                break
    assert plain_pos and other_pos
    a_in.pos = plain_pos
    a_out.pos = other_pos
    # Force food_mask vide pour éviter l'incertitude du manger
    env._food_mask[:] = False  # noqa: SLF001
    # Bloquer toute reproduction pendant le test (force last_repro_tick récent)
    for a in env._agents:  # noqa: SLF001
        a.last_repro_tick = 0
        a.last_build_tick = 0
    e_in_before = a_in.energy = 100.0  # < repro_threshold pour double-safety
    e_out_before = a_out.energy = 100.0
    # Step avec action 0 (NORTH) — bordures clamp
    env.step({a.agent_id: 0 for a in env._agents if a.alive})  # noqa: SLF001
    # Agent dans son biome (affinity=0=PLAIN tile=0) : metabolism × 0.7
    # Agent hors son biome (affinity=0 tile=autre) : metabolism × 1.5
    # Donc a_in.energy > a_out.energy
    assert a_in.energy > a_out.energy
