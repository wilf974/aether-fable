"""Tests pour V5 — construction de nids + rest bonus + cleanup à la mort."""
from __future__ import annotations

import pytest

from aetherlife.world.construction import BuildConfig, NestRecord
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


# ─── BuildConfig ──────────────────────────────────────────────────────────


def test_build_config_defaults() -> None:
    cfg = BuildConfig()
    assert cfg.enabled is False  # compat V2/V3/V4 préservée
    assert cfg.energy_threshold == 90.0
    assert cfg.build_cost == 25.0
    assert cfg.rest_bonus == 3.0
    assert cfg.cooldown_ticks == 50


def test_build_config_validates() -> None:
    with pytest.raises(ValueError):
        BuildConfig(energy_threshold=0)
    with pytest.raises(ValueError):
        BuildConfig(build_cost=0)
    with pytest.raises(ValueError):
        BuildConfig(build_cost=100, energy_threshold=50)
    with pytest.raises(ValueError):
        BuildConfig(rest_bonus=-1)
    with pytest.raises(ValueError):
        BuildConfig(cooldown_ticks=-1)


# ─── Compat V2/V3/V4 : par défaut, pas de construction ───────────────────


def test_v2_compat_no_build_by_default() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=2, max_energy=300.0, start_energy=250.0,
        food_value=20.0, initial_food_density=0.2, food_respawn_lambda=1.0,
        max_steps=30,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for _ in range(20):
        env.step({aid: 0 for aid in env.alive_agent_ids})
    assert env.n_nests == 0
    assert env.nests == {}


# ─── Construction activée ─────────────────────────────────────────────────


def test_build_triggers_when_energy_above_threshold() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=1, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=100,
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=30.0,
            rest_bonus=5.0, cooldown_ticks=1,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0})  # 1 step, agent doit construire
    assert env.n_nests == 1
    nest = list(env.nests.values())[0]
    assert nest.owner_id == 0
    assert nest.built_tick == 1


def test_rest_bonus_applied_on_own_nest() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=1, max_energy=200.0, start_energy=150.0,
        metabolism=1.0, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=100,
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=20.0,
            rest_bonus=10.0, cooldown_ticks=1,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    # Step 1 : agent reste à sa position (NORTH bloqué par bord), construit
    env._agents[0].pos = (0, 0)  # noqa: SLF001 — fix pos
    env.step({0: 0})  # NORTH au bord → reste (0,0), construit nid
    assert env.n_nests == 1
    energy_after_build = env.agent_state(0).energy
    # Step 2 : agent reste sur son nid → rest_bonus s'applique
    env.step({0: 0})  # NORTH bloqué, reste sur (0,0)
    energy_after_rest = env.agent_state(0).energy
    # Sans rest_bonus on perdrait metabolism, mais ici on récupère +10 -1 = +9
    assert energy_after_rest > energy_after_build - 1.0


def test_build_costs_energy() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=1, max_energy=200.0, start_energy=150.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=10,
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=30.0,
            rest_bonus=0.0, cooldown_ticks=1,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    initial_e = env.agent_state(0).energy
    env.step({0: 0})
    after_e = env.agent_state(0).energy
    # Construction coûte 30 + metabolism 0.1 = 30.1
    assert after_e < initial_e - 25.0


def test_at_most_one_nest_per_agent() -> None:
    """I16 — un agent ne peut avoir qu'un seul nid."""
    cfg = MultiAgentForagerConfig(
        rows=8, cols=8, n_agents=1, max_energy=300.0, start_energy=280.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=100,
        build=BuildConfig(
            enabled=True, energy_threshold=80.0, build_cost=20.0,
            rest_bonus=0.0, cooldown_ticks=1,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for _ in range(50):
        env.step({0: 0})
    assert env.n_nests <= 1


def test_nest_cleanup_on_owner_death() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=1, max_energy=200.0, start_energy=180.0,
        metabolism=1.0, food_value=10.0, death_penalty=0.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=300,
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=30.0,
            rest_bonus=0.0, cooldown_ticks=1,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0})  # construit
    assert env.n_nests == 1
    # Force la mort
    env._agents[0].energy = 0.5  # noqa: SLF001
    env.step({0: 0})
    assert env.n_alive == 0
    assert env.n_nests == 0  # nid retiré à la mort


def test_builds_last_step_tracking() -> None:
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=2, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=100,
        build=BuildConfig(
            enabled=True, energy_threshold=80.0, build_cost=20.0,
            rest_bonus=0.0, cooldown_ticks=1,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0, 1: 0})
    builds = env.builds_last_step
    assert len(builds) == 2  # les 2 agents construisent
    assert all(isinstance(b, NestRecord) for b in builds)
    # Step suivant : cooldown, pas de nouvelle construction
    env.step({0: 0, 1: 0})
    assert env.builds_last_step == []


def test_food_does_not_spawn_on_nest() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=1, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=5.0,  # spawn agressif
        max_steps=50,
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=20.0,
            rest_bonus=0.0, cooldown_ticks=1,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0})
    nest = list(env.nests.values())[0]
    # Run plusieurs steps, food ne doit jamais apparaître sur le nid
    for _ in range(20):
        env.step({0: 0})
        assert not env.food_mask[nest.pos[0], nest.pos[1]]


def test_nest_positions_set() -> None:
    """Les positions des nids sont uniques et correspondent à la position de leur owner."""
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=3, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=10,
        build=BuildConfig(
            enabled=True, energy_threshold=80.0, build_cost=20.0,
            rest_bonus=0.0, cooldown_ticks=1,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    # V6.3 : capture positions avant step (construction est faite avant move)
    start_positions = {
        a.agent_id: a.pos for a in env._agents if a.alive  # noqa: SLF001
    }
    env.step({0: 0, 1: 0, 2: 0})
    # Au moins 1 nid construit, max n_agents, positions uniques (pas de collision)
    assert 1 <= env.n_nests <= 3
    assert len(env.nest_positions) == env.n_nests  # toutes positions uniques
    # V6.3 : le nid est sur la position de départ de l'owner (avant son
    # mouvement du même tick, pas sa position finale)
    for nest in env.nests.values():
        assert nest.pos == start_positions[nest.owner_id]


# ─── Seasonal V5 ──────────────────────────────────────────────────────────


def test_seasonal_construction_works() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=6, cols=6, n_agents=2, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=30,
        seasonal=SeasonalConfig(season_period=30),
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=20.0,
            rest_bonus=5.0, cooldown_ticks=1,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0, 1: 0})
    assert env.n_nests >= 1


def test_seasonal_v3_compat_no_build_default() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=6, cols=6, n_agents=3, initial_food_density=0.2,
        food_respawn_lambda=1.0, max_steps=20,
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for _ in range(15):
        env.step({aid: 0 for aid in env.alive_agent_ids})
    assert env.n_nests == 0


def test_combined_reproduction_and_build() -> None:
    """V4 + V5 simultanés : agents bien nourris se reproduisent ET construisent."""
    from aetherlife.world.reproduction import ReproductionConfig

    cfg = MultiAgentForagerConfig(
        rows=10, cols=10, n_agents=2, max_energy=300.0, start_energy=250.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=200,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=120.0, energy_cost=60.0,
            cooldown_ticks=5, max_population=20,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=30.0,
            rest_bonus=2.0, cooldown_ticks=20,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    for _ in range(30):
        env.step({aid: 0 for aid in env.alive_agent_ids})
    assert env.n_births_total >= 1
    assert env.n_nests >= 1
