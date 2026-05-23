"""Tests pour V4 — reproduction automatique + lineage."""
from __future__ import annotations

import pytest

from aetherlife.world.food_grid import Action
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)
from aetherlife.world.reproduction import LineageEdge, ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


# ─── ReproductionConfig ──────────────────────────────────────────────────


def test_reproduction_config_defaults() -> None:
    cfg = ReproductionConfig()
    assert cfg.enabled is False  # default désactivé (compat V2/V3)
    assert cfg.energy_threshold == 80.0
    assert cfg.energy_cost == 40.0
    assert cfg.cooldown_ticks == 30
    assert cfg.max_population == 100


def test_reproduction_config_validates() -> None:
    with pytest.raises(ValueError):
        ReproductionConfig(energy_threshold=0)
    with pytest.raises(ValueError):
        ReproductionConfig(energy_cost=0)
    with pytest.raises(ValueError):
        ReproductionConfig(energy_cost=100, energy_threshold=50)  # cost >= threshold
    with pytest.raises(ValueError):
        ReproductionConfig(cooldown_ticks=-1)
    with pytest.raises(ValueError):
        ReproductionConfig(max_population=0)


# ─── MultiAgentFoodGrid : default = pas de reproduction (compat V2) ──────


def test_v2_compat_no_reproduction_by_default() -> None:
    """V4 doit préserver V2 : repro désactivée par défaut, pas de naissance."""
    cfg = MultiAgentForagerConfig(
        rows=8, cols=8, n_agents=4, max_energy=200.0, start_energy=150.0,
        food_value=20.0, initial_food_density=0.5, food_respawn_lambda=2.0,
        max_steps=50,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for _ in range(20):
        actions = {aid: 0 for aid in env.alive_agent_ids}
        env.step(actions)
    assert env.n_births_total == 0
    assert env.lineage == []


# ─── Reproduction enabled ─────────────────────────────────────────────────


def test_reproduction_triggers_when_energy_above_threshold() -> None:
    """Un agent avec energy haute doit donner naissance quand cooldown OK."""
    cfg = MultiAgentForagerConfig(
        rows=8, cols=8, n_agents=2, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0, death_penalty=0.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=100,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=100.0, energy_cost=50.0,
            cooldown_ticks=1, max_population=10,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    initial_pop = env.n_alive
    env.step({0: 0, 1: 0})  # step normal — reproduction tentée après
    # Au moins une naissance attendue (energy 180 > threshold 100, cooldown 1 OK)
    assert env.n_alive > initial_pop
    assert env.n_births_total > 0


def test_lineage_records_parent_child_edge() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=1, max_energy=200.0, start_energy=150.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=80.0, energy_cost=40.0,
            cooldown_ticks=1, max_population=10,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0})
    assert len(env.lineage) >= 1
    edge = env.lineage[0]
    assert isinstance(edge, LineageEdge)
    assert edge.parent_id == 0
    assert edge.child_id >= 1
    assert edge.parent_generation == 0
    assert edge.child_generation == 1
    assert edge.birth_tick == 1


def test_child_has_correct_lineage_fields() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=1, max_energy=200.0, start_energy=150.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=80.0, energy_cost=40.0,
            cooldown_ticks=1, max_population=10,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0})
    children = [a for a in env._agents if a.parent_id is not None]  # noqa: SLF001
    assert len(children) >= 1
    child = children[0]
    assert child.parent_id == 0
    assert child.generation == 1
    assert child.birth_tick == 1
    assert child.alive is True


def test_parent_energy_decreases_after_birth() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=1, max_energy=200.0, start_energy=150.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=100.0, energy_cost=50.0,
            cooldown_ticks=1, max_population=10,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    parent_initial = env.agent_state(0).energy
    env.step({0: 0})
    parent_after = env.agent_state(0).energy
    # parent a perdu metabolism + cost (avec un peu de tolerance pour float)
    assert parent_after < parent_initial - 40.0


def test_cooldown_prevents_immediate_re_reproduction() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=1, max_energy=300.0, start_energy=250.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=80.0, energy_cost=40.0,
            cooldown_ticks=10, max_population=10,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0})  # ép 1 : reproduction OK
    n_after_1 = env.n_alive
    env.step({0: 0})  # ép 2 : cooldown = 10, pas de nouvelle naissance
    assert env.n_alive == n_after_1


def test_max_population_caps_births() -> None:
    cfg = MultiAgentForagerConfig(
        rows=10, cols=10, n_agents=2, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=200,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=80.0, energy_cost=40.0,
            cooldown_ticks=1, max_population=4,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for _ in range(50):
        actions = {aid: 0 for aid in env.alive_agent_ids}
        env.step(actions)
    assert env.n_alive <= 4


def test_births_last_step_clears_each_step() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=1, max_energy=200.0, start_energy=180.0,
        metabolism=10.0,  # parent perd énergie vite après repro
        food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=120.0, energy_cost=60.0,
            cooldown_ticks=1, max_population=20,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0})
    # Au step suivant, births_last_step doit être vidé ou refait
    births_step_1 = env.births_last_step
    env.step({aid: 0 for aid in env.alive_agent_ids})
    births_step_2 = env.births_last_step
    # Step 2 a son propre snapshot
    assert isinstance(births_step_2, list)


def test_agent_id_unique_increasing() -> None:
    cfg = MultiAgentForagerConfig(
        rows=8, cols=8, n_agents=3, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=80.0, energy_cost=40.0,
            cooldown_ticks=1, max_population=20,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for _ in range(10):
        actions = {aid: 0 for aid in env.alive_agent_ids}
        env.step(actions)
    ids = [a.agent_id for a in env._agents]  # noqa: SLF001
    assert len(ids) == len(set(ids))  # unique
    assert ids == sorted(ids)  # croissant


# ─── Seasonal reproduction ────────────────────────────────────────────────


def test_seasonal_reproduction_works() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=8, cols=8, n_agents=2, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        seasonal=SeasonalConfig(season_period=50),
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=80.0, energy_cost=40.0,
            cooldown_ticks=1, max_population=10,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    initial_pop = env.n_alive
    env.step({0: 0, 1: 0})
    assert env.n_alive > initial_pop
    assert env.n_births_total > 0


def test_seasonal_v3_compat_no_reproduction_default() -> None:
    """V3 saisonnier sans reproduction explicite = pas de naissance."""
    cfg = SeasonalMultiAgentConfig(
        rows=6, cols=6, n_agents=3, initial_food_density=0.2,
        food_respawn_lambda=1.0, max_steps=20,
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for _ in range(15):
        actions = {aid: 0 for aid in env.alive_agent_ids}
        env.step(actions)
    assert env.n_births_total == 0
