"""Tests V5.3 — caches food (deposit / withdrawal / family / cleanup)."""
from __future__ import annotations

import pytest

from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)


# ─── CacheConfig validation ───────────────────────────────────────────────


def test_cache_config_defaults() -> None:
    cfg = CacheConfig()
    assert cfg.enabled is False
    assert cfg.deposit_threshold == 120.0
    assert cfg.withdrawal_threshold == 40.0
    assert cfg.max_capacity == 60.0
    assert cfg.deposit_amount == 5.0
    assert cfg.withdrawal_amount == 5.0


def test_cache_config_validates() -> None:
    with pytest.raises(ValueError):
        CacheConfig(deposit_threshold=0)
    with pytest.raises(ValueError):
        CacheConfig(withdrawal_threshold=-1)
    with pytest.raises(ValueError):
        CacheConfig(deposit_threshold=40, withdrawal_threshold=50)
    with pytest.raises(ValueError):
        CacheConfig(max_capacity=0)
    with pytest.raises(ValueError):
        CacheConfig(deposit_amount=0)
    with pytest.raises(ValueError):
        CacheConfig(withdrawal_amount=0)


# ─── Compat V5.0 / V5.2 : cache OFF par défaut ────────────────────────────


def test_no_cache_when_disabled() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=1, max_energy=200.0, start_energy=180.0,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=50,
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=20.0,
            rest_bonus=5.0, cooldown_ticks=1,
        ),
        # cache désactivé par default
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env._agents[0].pos = (0, 0)  # noqa: SLF001
    for _ in range(10):
        env.step({0: 0})
    assert env.nest_food_stock == {}
    assert env.cache_deposits_total == 0
    assert env.total_cached_food == 0


# ─── Deposit / withdrawal ─────────────────────────────────────────────────


def _make_cache_env(
    *, family: bool = False, max_energy: float = 200.0, start_energy: float = 180.0,
    rest_bonus: float = 0.0,
) -> MultiAgentFoodGrid:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=1, max_energy=max_energy, start_energy=start_energy,
        metabolism=0.1, food_value=10.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=100,
        build=BuildConfig(
            enabled=True, energy_threshold=100.0, build_cost=20.0,
            rest_bonus=rest_bonus, cooldown_ticks=1,
            family_inheritance=family,
        ),
        cache=CacheConfig(
            enabled=True, deposit_threshold=120.0, withdrawal_threshold=40.0,
            max_capacity=50.0, deposit_amount=5.0, withdrawal_amount=5.0,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env._agents[0].pos = (0, 0)  # noqa: SLF001
    return env


def test_deposit_when_above_threshold() -> None:
    env = _make_cache_env(start_energy=180.0)
    # Step 1 : construit le nid (start_energy=180 > build_threshold=100)
    env.step({0: 0})
    assert env.n_nests == 1
    # Step 2 : agent toujours sur (0,0) → dépose dans le cache
    e_before = env.agent_state(0).energy
    env.step({0: 0})
    assert env.cache_deposits_total >= 1
    assert env.total_cached_food > 0
    # Énergie de l'agent diminue (deposit prélevé)
    assert env.agent_state(0).energy < e_before


def test_withdrawal_when_low_energy() -> None:
    env = _make_cache_env(start_energy=180.0)
    env.step({0: 0})  # build nest
    # Charger le cache pendant 10 steps (180 → ~130)
    for _ in range(10):
        env.step({0: 0})
    assert env.total_cached_food > 0
    cache_before = env.total_cached_food
    # Force énergie basse
    env._agents[0].energy = 30.0  # noqa: SLF001
    e_before = env.agent_state(0).energy
    env.step({0: 0})
    # Agent a tiré du cache
    assert env.cache_withdrawals_total >= 1
    assert env.total_cached_food < cache_before
    assert env.agent_state(0).energy > e_before  # énergie regagnée


def test_cache_capped_at_max_capacity() -> None:
    env = _make_cache_env(start_energy=180.0, max_energy=500.0)
    env.step({0: 0})  # build
    # Beaucoup de steps pour saturer le cache
    for _ in range(50):
        env._agents[0].energy = 150.0  # noqa: SLF001 — force au-dessus du deposit_threshold
        env.step({0: 0})
    assert env.total_cached_food <= 50.0  # max_capacity


def test_cache_lost_on_death_v5() -> None:
    """V5.0 : cache perdu à la mort du builder.

    Note importante (finding V5.3) : le cache **sauve** un agent quand
    energy < withdrawal_threshold ET cache > 0. Pour tester le cleanup à
    la mort, on doit donc vider le cache OU forcer une énergie tellement
    basse qu'aucun withdrawal ne peut sauver l'agent.
    """
    env = _make_cache_env(family=False)
    env.step({0: 0})  # build
    for _ in range(5):
        env.step({0: 0})
    assert env.total_cached_food > 0
    # Vider le cache manuellement pour empêcher la sauvegarde
    env._nest_food_stock[0] = 0.0  # noqa: SLF001
    # Énergie suffisamment basse pour mourir même avec metabolism
    env._agents[0].energy = -1.0  # noqa: SLF001
    env.step({0: 0})
    assert env.n_alive == 0
    assert env.n_nests == 0
    assert env.total_cached_food == 0  # cache perdu (vidé manuellement + cleanup)


def test_cache_can_save_low_energy_agent() -> None:
    """V5.3 finding : le cache permet à un agent affamé de survivre."""
    env = _make_cache_env(family=False)
    env.step({0: 0})  # build
    # Charger le cache pendant plusieurs steps
    for _ in range(10):
        env.step({0: 0})
    assert env.total_cached_food > 0
    # Force énergie très basse (mais positive)
    env._agents[0].energy = 1.0  # noqa: SLF001
    env.step({0: 0})
    # L'agent a tiré du cache et survécu
    assert env.n_alive == 1
    assert env.cache_withdrawals_total >= 1
    assert env.agent_state(0).energy > 1.0  # énergie a augmenté


def test_no_deposit_below_threshold() -> None:
    env = _make_cache_env(start_energy=180.0)
    env.step({0: 0})  # build
    # Force énergie pile au seuil rest mais sous deposit
    env._agents[0].energy = 100.0  # noqa: SLF001 — entre withdrawal_threshold (40) et deposit (120)
    env.step({0: 0})
    # Pas de dépôt ni retrait (zone neutre)
    assert env.cache_deposits_total == 0 or env.total_cached_food == 0


def test_no_withdrawal_when_cache_empty() -> None:
    env = _make_cache_env(start_energy=180.0)
    env.step({0: 0})  # build
    # Force énergie basse direct, cache vide
    env._agents[0].energy = 20.0  # noqa: SLF001
    e_before = env.agent_state(0).energy
    env.step({0: 0})
    assert env.cache_withdrawals_total == 0
    assert env.total_cached_food == 0


def test_cache_persists_with_family_when_descendant_alive() -> None:
    """V5.2 + V5.3 : cache survit à la mort du parent si un descendant vit."""
    from aetherlife.world.reproduction import ReproductionConfig

    cfg = MultiAgentForagerConfig(
        rows=6, cols=6, n_agents=1, max_energy=300.0, start_energy=280.0,
        metabolism=0.1, food_value=10.0, death_penalty=0.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=300,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=150.0, energy_cost=70.0,
            cooldown_ticks=2, max_population=10,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=120.0, build_cost=20.0,
            rest_bonus=0.0, cooldown_ticks=2, family_inheritance=True,
        ),
        cache=CacheConfig(
            enabled=True, deposit_threshold=150.0, withdrawal_threshold=40.0,
            max_capacity=50.0, deposit_amount=10.0, withdrawal_amount=5.0,
        ),
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env._agents[0].pos = (0, 0)  # noqa: SLF001
    # Run beaucoup de steps pour avoir nid + cache + enfants
    for _ in range(30):
        env.step({aid: 0 for aid in env.alive_agent_ids})
    if env.total_cached_food == 0 or env.n_alive < 2:
        pytest.skip("setup probabiliste : le cache ou les enfants n'existent pas")
    cache_initial = env.total_cached_food
    # Tuer le parent (id 0) ; enfants partagent root_ancestor_id=0
    env._agents[0].energy = 0.5  # noqa: SLF001
    env.step({aid: 0 for aid in env.alive_agent_ids})
    # Si un descendant vit, le nid (et son cache) persiste
    if env.has_living_descendant(0):
        assert 0 in env.nest_food_stock or env.total_cached_food >= cache_initial - 10
