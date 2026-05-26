"""Tests V8-B1.2 — LineageAgent : policy RL multi-agent + observation égocentrique."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aetherlife.agents.lineage_agent import (
    LineageAgent, egocentric_obs, egocentric_obs_dim,
)
from aetherlife.agents.lineage_brain import BrainConfig
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig, SeasonalMultiAgentConfig, SeasonalMultiAgentFoodGrid,
)


def _make_env(n_agents: int = 4, rows: int = 12, cols: int = 12) -> SeasonalMultiAgentFoodGrid:
    cfg = SeasonalMultiAgentConfig(
        rows=rows, cols=cols, n_agents=n_agents,
        max_energy=200.0, start_energy=140.0,
        metabolism=0.4, food_value=15.0, death_penalty=0.0,
        initial_food_density=0.1, food_respawn_lambda=0.5, max_steps=500,
        seasonal=SeasonalConfig(season_period=100),
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=120.0, energy_cost=60.0,
            cooldown_ticks=20, max_population=10,
        ),
        build=BuildConfig(enabled=True, energy_threshold=100.0, build_cost=20.0,
                          rest_bonus=3.0, cooldown_ticks=10),
        cache=CacheConfig(enabled=False),
        planting=PlantingConfig(enabled=False),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    return env


def test_egocentric_obs_dim() -> None:
    # V8-B1.5 : 5 canaux (food, nests, plants, agents, biome) + 3 scalars
    assert egocentric_obs_dim(5) == 5 * 121 + 3  # 608
    assert egocentric_obs_dim(3) == 5 * 49 + 3   # 248
    assert egocentric_obs_dim(1) == 5 * 9 + 3    # 48


def test_egocentric_obs_shape() -> None:
    env = _make_env(n_agents=2)
    agent = env._agents[0]  # noqa: SLF001
    obs = egocentric_obs(env, agent, vision_radius=5)
    assert obs.shape == (608,)
    assert obs.dtype == np.float32


def test_egocentric_obs_self_excluded() -> None:
    """L'agent ne se voit pas lui-même dans le canal agents."""
    env = _make_env(n_agents=1)
    agent = env._agents[0]  # noqa: SLF001
    obs = egocentric_obs(env, agent, vision_radius=2)
    size = 5
    # canal agents = 4ème (index 3) → offset 3 * size² .. 4 * size²
    agent_view = obs[3 * size * size: 4 * size * size].reshape(size, size)
    # Le centre (r, r) = position de l'agent ; doit être 0 (pas lui-même)
    assert agent_view[2, 2] == 0.0


def test_egocentric_obs_embedding_dim_pad_when_no_vocab() -> None:
    """V8-C3 P1 régression : si vocab actif (embedding_dim>0) MAIS
    listener_vocab absent, l'obs doit padder à zéro pour rester de dim
    constante. Sinon le replay buffer crashe (489 vs 505)."""
    env = _make_env(n_agents=2)
    agent = env._agents[0]  # noqa: SLF001
    # Cas 1 : listener_vocab=None, embedding_dim=0 → pas de heard_vec
    obs_no_vocab = egocentric_obs(env, agent, vision_radius=5)
    assert obs_no_vocab.shape == (5 * 121 + 3,)  # 608
    # Cas 2 : listener_vocab=None, embedding_dim=16 → padding zéro
    obs_padded = egocentric_obs(
        env, agent, vision_radius=5, embedding_dim=16,
    )
    assert obs_padded.shape == (5 * 121 + 3 + 16,)  # 624
    # Le bloc de padding doit être zéro
    assert np.all(obs_padded[-16:] == 0.0)


def test_egocentric_obs_auto_infer_embedding_from_env() -> None:
    """V8-C3 P1 fix v2 : si env.cfg.vocabulary.enabled mais listener_vocab
    et embedding_dim sont absents, egocentric_obs doit auto-padder depuis
    env.cfg.vocabulary.embedding_dim. Couvre les call sites qui n'ont pas
    accès au LineageAgent (viz, bench)."""
    from aetherlife.world.vocabulary import VocabularyConfig
    cfg = SeasonalMultiAgentConfig(
        rows=10, cols=10, n_agents=2,
        max_energy=200.0, start_energy=140.0,
        metabolism=0.4, food_value=15.0, death_penalty=0.0,
        initial_food_density=0.1, food_respawn_lambda=0.5, max_steps=500,
        seasonal=SeasonalConfig(season_period=100),
        reproduction=ReproductionConfig(enabled=False),
        build=BuildConfig(enabled=False),
        cache=CacheConfig(enabled=False),
        planting=PlantingConfig(enabled=False),
        vocabulary=VocabularyConfig(
            enabled=True, n_tokens=4, embedding_dim=16,
            listen_radius=5,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    agent = env._agents[0]  # noqa: SLF001
    # Aucun param vocab passé, mais env.cfg.vocabulary.enabled=True
    obs = egocentric_obs(env, agent, vision_radius=5)
    # Dim attendue : 5 channels × 121 + 3 + 16 (auto-padding)
    assert obs.shape == (5 * 121 + 3 + 16,)
    assert np.all(obs[-16:] == 0.0)


def test_egocentric_obs_values_in_range() -> None:
    env = _make_env(n_agents=3)
    agent = env._agents[0]  # noqa: SLF001
    obs = egocentric_obs(env, agent, vision_radius=5)
    # Canaux binaires (4 premiers)
    binary_part = obs[: 4 * 121]
    assert np.all((binary_part == 0.0) | (binary_part == 1.0))
    # Canal biome (5ème) : valeurs ∈ [0, 1]
    biome_part = obs[4 * 121: 5 * 121]
    assert np.all((biome_part >= 0.0) & (biome_part <= 1.0))
    # Scalars normalisés
    e, age, phase = obs[-3], obs[-2], obs[-1]
    assert 0.0 <= e <= 1.0
    assert 0.0 <= age <= 1.0
    assert 0.0 <= phase < 1.0


def test_lineage_agent_creates_brains_per_lineage() -> None:
    env = _make_env(n_agents=3)
    cfg = BrainConfig(enabled=True, device="cpu", vision_radius=3)
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=0)
    # Chaque agent fondateur a son propre brain (root = own id)
    assert len(policy.registry) == 3
    for a in env._agents:  # noqa: SLF001
        assert a.root_ancestor_id in policy.registry


def test_lineage_agent_act_dict_returns_valid_actions() -> None:
    env = _make_env(n_agents=4)
    cfg = BrainConfig(enabled=True, device="cpu", vision_radius=3)
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=0)
    obs_dict = {a.agent_id: np.zeros(10) for a in env._agents}  # noqa: SLF001
    actions = policy.act_dict(obs_dict, greedy=True)
    assert len(actions) == 4
    for a in actions.values():
        assert 0 <= a < 4


def test_lineage_agent_observes_and_learns() -> None:
    env = _make_env(n_agents=2)
    cfg = BrainConfig(
        enabled=True, device="cpu", vision_radius=3,
        min_replay_to_learn=10, batch_size=4, buffer_capacity=100,
    )
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=0)
    obs_dict = {a.agent_id: np.zeros(10) for a in env._agents}  # noqa: SLF001
    actions = policy.act_dict(obs_dict, greedy=False)
    # Construire des obs égocentriques
    prev_ego = {
        aid: egocentric_obs(env, env.agent_state(aid), 3)
        for aid in actions
    }
    rewards = {aid: 0.1 for aid in actions}
    dones = {aid: False for aid in actions}
    root_ids = {aid: env.agent_state(aid).root_ancestor_id for aid in actions}
    metrics = policy.observe_dict(
        prev_obs_ego=prev_ego,
        actions=actions,
        rewards=rewards,
        next_obs_ego=prev_ego,  # même obs pour test
        dones=dones,
        agent_root_ids=root_ids,
    )
    # Steps incrémentés
    for brain in policy.registry:
        assert brain.global_step == 1
    assert "mean_epsilon" in metrics


def test_lineage_agent_smoke_50_ticks_no_crash() -> None:
    """Smoke : 50 ticks sans erreur, lignées peuvent s'éteindre."""
    env = _make_env(n_agents=3, rows=8, cols=8)
    cfg = BrainConfig(
        enabled=True, device="cpu", vision_radius=3,
        min_replay_to_learn=4, batch_size=4, buffer_capacity=200,
    )
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=0)
    obs = {a.agent_id: np.zeros(10) for a in env._agents}  # noqa: SLF001
    for _ in range(50):
        if env.n_alive == 0:
            break
        actions = policy.act_dict(obs, greedy=False)
        env.step(actions)
    # Pas de crash, registry contient ≥ 0 brains
    assert len(policy.registry) >= 0
