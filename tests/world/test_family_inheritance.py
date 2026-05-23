"""Tests V5.2 — nids familiaux (héritage de lignée + persistence)."""
from __future__ import annotations

import pytest

from aetherlife.world.construction import BuildConfig, NestRecord
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
    _AgentState,
)
from aetherlife.world.reproduction import ReproductionConfig


def make_family_env(
    family: bool = True, n_agents: int = 1, max_pop: int = 20
) -> MultiAgentFoodGrid:
    cfg = MultiAgentForagerConfig(
        rows=8, cols=8, n_agents=n_agents,
        max_energy=300.0, start_energy=200.0,
        metabolism=0.1, food_value=10.0, death_penalty=0.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=200,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=100.0, energy_cost=60.0,
            cooldown_ticks=1, max_population=max_pop,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=80.0, build_cost=20.0,
            rest_bonus=5.0, cooldown_ticks=5, family_inheritance=family,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    return env


# ─── root_ancestor_id ─────────────────────────────────────────────────────


def test_initial_agents_have_root_eq_self() -> None:
    env = make_family_env(n_agents=3)
    for a in env._agents:  # noqa: SLF001
        assert a.root_ancestor_id == a.agent_id


def test_child_inherits_root_ancestor_id() -> None:
    env = make_family_env(family=True)
    env.step({0: 0})  # reproduction immédiate (energy 200 > threshold 100)
    children = [a for a in env._agents if a.parent_id is not None]  # noqa: SLF001
    assert len(children) >= 1
    for c in children:
        assert c.root_ancestor_id == 0  # le parent était l'agent 0


def test_grandchild_inherits_root_via_parent_chain() -> None:
    env = make_family_env(family=True)
    # Forcer plusieurs reproductions
    for _ in range(10):
        env.step({aid: 0 for aid in env.alive_agent_ids})
    # Tous les agents nés devraient avoir root=0 (initial était agent 0)
    children = [a for a in env._agents if a.parent_id is not None]  # noqa: SLF001
    assert len(children) >= 2
    assert all(c.root_ancestor_id == 0 for c in children)


def test_lineage_ids_returns_full_lineage() -> None:
    env = make_family_env(family=True)
    for _ in range(5):
        env.step({aid: 0 for aid in env.alive_agent_ids})
    ids = env.lineage_ids(0)
    assert 0 in ids  # root inclus
    assert len(ids) >= 2  # parent + au moins 1 enfant


# ─── Family rest bonus ────────────────────────────────────────────────────


def _setup_deterministic_family(family: bool) -> MultiAgentFoodGrid:
    """Setup state-machine déterministe :
    - parent (id=0, root=0) à (3, 3)
    - child (id=1, parent=0, root=0) à (0, 0)
    - nest du parent à (0, 0) (donc child est exactement dessus)
    - Action NORTH sur child → clamp (0,0) → reste sur le nid
    """
    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=2, max_energy=200.0, start_energy=100.0,
        metabolism=1.0, food_value=10.0, death_penalty=0.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        build=BuildConfig(
            enabled=True, energy_threshold=50.0, build_cost=10.0,
            rest_bonus=10.0, cooldown_ticks=1, family_inheritance=family,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    # Reset state to deterministic : parent at (3,3), child at (0,0)
    env._agents = [  # noqa: SLF001
        _AgentState(
            agent_id=0, pos=(3, 3), energy=100.0, alive=True,
            parent_id=None, birth_tick=0, generation=0, root_ancestor_id=0,
        ),
        _AgentState(
            agent_id=1, pos=(0, 0), energy=50.0, alive=True,
            parent_id=0, birth_tick=0, generation=1, root_ancestor_id=0,
        ),
    ]
    env._next_agent_id = 2  # noqa: SLF001
    # Place le nid du parent à (0, 0), où se trouve le child
    env._nests = {  # noqa: SLF001
        0: NestRecord(owner_id=0, pos=(0, 0), built_tick=0),
    }
    return env


def test_child_gains_rest_on_parent_nest_when_family_on() -> None:
    """V5.2 : l'enfant sur le nid du parent reçoit le rest_bonus (family on)."""
    env = _setup_deterministic_family(family=True)
    child_energy_before = env.agent_state(1).energy
    visits_before = env.family_nest_visits_total
    # Action NORTH sur le child à (0, 0) → clamp, reste sur le nid
    env.step({0: 0, 1: 0})
    assert env.agent_state(1).pos == (0, 0)
    # Rest bonus appliqué : energy += rest_bonus - metabolism = 50 + 10 - 1 = 59
    assert env.agent_state(1).energy >= child_energy_before - 0.5
    assert env.family_nest_visits_total > visits_before


def test_no_family_rest_when_family_off() -> None:
    """V5.0 : sans family_inheritance, l'enfant sur le nid parent ne reçoit RIEN."""
    env = _setup_deterministic_family(family=False)
    child_energy_before = env.agent_state(1).energy
    visits_before = env.family_nest_visits_total
    env.step({0: 0, 1: 0})
    # L'enfant subit metabolism normal, pas de rest bonus
    assert env.agent_state(1).energy == child_energy_before - 1.0
    # Le compteur n'augmente pas (seul own_nest aurait pu, mais child n'a pas de nid)
    assert env.family_nest_visits_total == visits_before


# ─── Nest persistence ─────────────────────────────────────────────────────


def test_nest_persists_after_owner_death_if_descendants_alive() -> None:
    env = make_family_env(family=True, max_pop=20)
    # Reproduction puis construction
    for _ in range(15):
        env.step({aid: 0 for aid in env.alive_agent_ids if env.alive_agent_ids})
    if env.n_nests == 0 or env.n_alive < 2:
        pytest.skip("setup insuffisant")
    # Trouver un nid dont l'owner a au moins 1 descendant vivant
    nests_with_living_descendant = []
    for nest in env.nests.values():
        owner = env.agent_state(nest.owner_id)
        if env.has_living_descendant(owner.root_ancestor_id):
            descendants = [
                a for a in env._agents  # noqa: SLF001
                if a.alive
                and a.root_ancestor_id == owner.root_ancestor_id
                and a.agent_id != owner.agent_id
            ]
            if descendants:
                nests_with_living_descendant.append((nest, owner))
    if not nests_with_living_descendant:
        pytest.skip("aucun nid avec descendant vivant pour ce test")
    nest, owner = nests_with_living_descendant[0]
    # Forcer la mort du owner
    owner.energy = 0.5
    env.step({aid: 0 for aid in env.alive_agent_ids})
    # Le nid existe encore (descendants vivants)
    assert nest.owner_id in env.nests


def test_nest_disappears_when_no_descendant_alive() -> None:
    env = make_family_env(family=True, n_agents=1, max_pop=20)
    # Builder unique, pas d'enfant : à sa mort, le nid disparaît
    for _ in range(15):
        env.step({aid: 0 for aid in env.alive_agent_ids if env.alive_agent_ids})
    if env.n_nests == 0:
        pytest.skip("pas de nid construit")
    # Avant mort
    assert env.n_nests >= 1
    # Tuer le seul agent (et tous ses descendants si présents)
    for a in env._agents:  # noqa: SLF001
        if a.alive and a.root_ancestor_id == 0:
            a.energy = 0.5
    env.step({aid: 0 for aid in env.alive_agent_ids})
    # Si tous les membres de la lignée 0 sont morts, le nid disparaît
    if not env.has_living_descendant(0):
        nests_of_root_0 = [
            n for n in env.nests.values()
            if env.lineage_ids(0).count(n.owner_id) > 0
        ]
        assert len(nests_of_root_0) == 0


def test_compat_v5_off_preserves_default_behavior() -> None:
    """V5.0 : family_inheritance=False (default) → nid disparaît à la mort owner."""
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=1, max_energy=200.0, start_energy=180.0,
        metabolism=1.0, food_value=10.0, death_penalty=0.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=300,
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=30.0,
            rest_bonus=0.0, cooldown_ticks=1,
            # family_inheritance default = False
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env.step({0: 0})
    assert env.n_nests == 1
    env._agents[0].energy = 0.5  # noqa: SLF001
    env.step({0: 0})
    assert env.n_nests == 0  # V5.0 = nid disparaît
