"""Tests V8-B1.7 — Seed bank + respawn extinct affinities."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aetherlife.agents.lineage_agent import LineageAgent
from aetherlife.agents.lineage_brain import BrainConfig, LineageBrain
from aetherlife.agents.lineage_registry import LineageRegistry
from aetherlife.world.biomes import BiomeConfig
from aetherlife.world.cache import CacheConfig
from aetherlife.world.competition import CompetitionConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig, SeasonalMultiAgentConfig, SeasonalMultiAgentFoodGrid,
)


def _cfg() -> BrainConfig:
    return BrainConfig(enabled=True, device="cpu")


# ─── LineageBrain stocke affinity ───────────────────────────────────────


def test_brain_stores_affinity() -> None:
    b = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=_cfg(),
        seed=0, biome_affinity=2,
    )
    assert b.biome_affinity == 2


def test_brain_inherit_propagates_affinity() -> None:
    parent = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=_cfg(),
        seed=0, biome_affinity=1,
    )
    child = LineageBrain.inherit_from(parent=parent, root_id=1, seed=1)
    assert child.biome_affinity == 1


def test_brain_inherit_can_override_affinity() -> None:
    parent = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=_cfg(),
        seed=0, biome_affinity=1,
    )
    child = LineageBrain.inherit_from(
        parent=parent, root_id=1, seed=1, biome_affinity=3,
    )
    assert child.biome_affinity == 3


# ─── Seed bank registry ──────────────────────────────────────────────────


def test_registry_archives_brain_on_cull() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    b = reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    b.biome_affinity = 2
    assert reg.seed_bank_size() == 0
    # Cull (immédiat car grace_ticks=0)
    reg.cull_dead_lineages(alive_roots=set())
    assert reg.seed_bank_size() == 1
    assert reg.seed_bank_size(affinity=2) == 1


def test_seed_bank_returns_archived() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    b = reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    b.biome_affinity = 1
    reg.cull_dead_lineages(alive_roots=set())
    retrieved = reg.get_seed_brain_for_affinity(1)
    assert retrieved is b


def test_seed_bank_capacity_per_affinity() -> None:
    """FIFO bornée à seed_bank_max_per_affinity."""
    reg = LineageRegistry(
        cfg=_cfg(), obs_dim=10, n_actions=4,
        seed_bank_max_per_affinity=2,
    )
    # Archive 3 brains de même affinity → FIFO garde les 2 plus récents
    for i in range(3):
        b = reg.get_or_create(root_id=i, parent_brain=None, seed=i)
        b.biome_affinity = 0
        reg.cull_dead_lineages(alive_roots=set())
    assert reg.seed_bank_size(affinity=0) == 2


def test_seed_bank_no_affinity_not_archived() -> None:
    """Brain sans affinity → pas archivé."""
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    reg.get_or_create(root_id=0, parent_brain=None, seed=0)  # affinity=None
    reg.cull_dead_lineages(alive_roots=set())
    assert reg.seed_bank_size() == 0


# ─── env.spawn_founder ──────────────────────────────────────────────────


def _make_env() -> SeasonalMultiAgentFoodGrid:
    cfg = SeasonalMultiAgentConfig(
        rows=12, cols=12, n_agents=4, max_energy=300.0, start_energy=180.0,
        metabolism=0.3, food_value=18.0, death_penalty=0.0,
        initial_food_density=0.1, food_respawn_lambda=0.5, max_steps=1000,
        seasonal=SeasonalConfig(season_period=100),
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=130.0, energy_cost=70.0,
            cooldown_ticks=20, max_population=20,
        ),
        build=BuildConfig(enabled=False),
        cache=CacheConfig(enabled=False),
        planting=PlantingConfig(enabled=False),
        biomes=BiomeConfig(
            enabled=True, n_seed_points=8, balanced_seeds=True,
            affinity_enabled=True, respawn_enabled=True,
        ),
        competition=CompetitionConfig(enabled=False),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    return env


def test_spawn_founder_creates_agent_in_biome() -> None:
    env = _make_env()
    n_before = len([a for a in env._agents if a.alive])  # noqa: SLF001
    new_id = env.spawn_founder(affinity=2)
    assert new_id is not None
    new_ag = env.agent_state(new_id)
    assert new_ag.biome_affinity == 2
    # Sa position est sur un tile de biome 2
    assert int(env._biome_map[new_ag.pos[0], new_ag.pos[1]]) == 2  # noqa: SLF001
    # Population a augmenté
    assert len([a for a in env._agents if a.alive]) == n_before + 1  # noqa: SLF001


def test_spawn_founder_returns_none_if_no_tiles() -> None:
    """Si le biome n'existe pas dans la map, spawn échoue."""
    env = _make_env()
    # Remplir le biome map avec 0 partout → biome 3 absent
    env._biome_map[:] = 0  # noqa: SLF001
    new_id = env.spawn_founder(affinity=3)
    assert new_id is None


# ─── LineageAgent.maybe_respawn_extinct_affinities ──────────────────────


def test_lineage_agent_respawn_after_extinction() -> None:
    env = _make_env()
    cfg = BrainConfig(
        enabled=True, device="cpu", vision_radius=3,
    )
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=0)
    # Tuer artificiellement tous les agents d'affinity 0
    for a in env._agents:  # noqa: SLF001
        if a.biome_affinity == 0:
            a.alive = False
    # Forcer le tick suffisamment loin pour déclencher respawn
    env._step_count = 6000  # noqa: SLF001 — > respawn_extinct_after_ticks
    n_respawned = policy.maybe_respawn_extinct_affinities()
    assert n_respawned >= 1
    # Il devrait y avoir au moins 1 agent vivant avec affinity=0
    n_aff0 = sum(
        1 for a in env._agents
        if a.alive and a.biome_affinity == 0  # noqa: SLF001
    )
    assert n_aff0 >= 1


def test_respawn_disabled_does_nothing() -> None:
    """Si respawn_enabled=False, pas de respawn."""
    cfg = SeasonalMultiAgentConfig(
        rows=10, cols=10, n_agents=4,
        biomes=BiomeConfig(
            enabled=True, affinity_enabled=True, respawn_enabled=False,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    brain_cfg = BrainConfig(enabled=True, device="cpu", vision_radius=3)
    policy = LineageAgent(env=env, cfg=brain_cfg, n_actions=4, seed=0)
    env._step_count = 10000  # noqa: SLF001
    n = policy.maybe_respawn_extinct_affinities()
    assert n == 0
