"""V8-B1 benchmark — DQN par lignée vs SmartHeuristic baseline.

Run 1k ticks en mode garden léger. Mesure :
    - survie cumulée
    - lifespan moyen
    - food eaten total
    - n births
    - n lignées survivantes
    - dominance (top lignée %)
    - loss moyenne (RL en train)
    - epsilon final
"""
from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

from aetherlife.agents.lineage_agent import LineageAgent, egocentric_obs
from aetherlife.agents.lineage_brain import BrainConfig
from aetherlife.agents.smart_heuristic import SmartHeuristicAgent
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig, SeasonalMultiAgentConfig, SeasonalMultiAgentFoodGrid,
)


def build_env(seed: int) -> SeasonalMultiAgentFoodGrid:
    cfg = SeasonalMultiAgentConfig(
        rows=24, cols=24, n_agents=8, max_energy=300.0, start_energy=160.0,
        metabolism=0.5, food_value=18.0, death_penalty=0.0,
        initial_food_density=0.06, food_respawn_lambda=0.25, max_steps=2000,
        seasonal=SeasonalConfig(season_period=200),
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=130.0, energy_cost=65.0,
            cooldown_ticks=80, max_population=20,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=110.0, build_cost=30.0,
            rest_bonus=4.0, cooldown_ticks=80, family_inheritance=True,
        ),
        cache=CacheConfig(enabled=False),
        planting=PlantingConfig(enabled=False),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    return env


def run_baseline(seed: int, n_ticks: int) -> dict:
    env = build_env(seed)
    policy = SmartHeuristicAgent(env, seed=seed)
    obs = {a.agent_id: np.zeros(10) for a in env._agents}  # noqa: SLF001
    t0 = time.time()
    for _ in range(n_ticks):
        if env.n_alive == 0:
            break
        actions = policy.act_dict(obs, greedy=True)
        env.step(actions)
    dt = time.time() - t0
    return _summary(env, dt, label="SmartHeuristic")


def run_lineage_dqn(seed: int, n_ticks: int, device: str) -> dict:
    env = build_env(seed)
    cfg = BrainConfig(
        enabled=True, device=device, vision_radius=4,
        hidden_dims=(64, 64), lr=5e-4, batch_size=64,
        buffer_capacity=10_000, min_replay_to_learn=200, train_every=4,
        epsilon_start=0.7, epsilon_end=0.05, epsilon_decay_steps=5_000,
        target_sync_steps=200, mutation_std=0.03,
    )
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=seed)
    obs_stub = {a.agent_id: np.zeros(10) for a in env._agents}  # noqa: SLF001
    t0 = time.time()

    last_ego: dict[int, np.ndarray] = {
        a.agent_id: egocentric_obs(env, a, cfg.vision_radius)
        for a in env._agents  # noqa: SLF001
        if a.alive
    }
    last_actions: dict[int, int] = {}
    last_roots: dict[int, int] = {
        a.agent_id: a.root_ancestor_id
        for a in env._agents  # noqa: SLF001
    }

    for t in range(n_ticks):
        if env.n_alive == 0:
            break
        # Refresh obs_stub with current alive
        obs_stub = {a.agent_id: np.zeros(10) for a in env._agents if a.alive}  # noqa: SLF001
        actions = policy.act_dict(obs_stub, greedy=False)
        # Save energy before pour calculer reward delta
        e_before = {
            a.agent_id: a.energy
            for a in env._agents if a.alive  # noqa: SLF001
        }
        env.step(actions)
        # Construire next obs égocentriques + rewards (Δenergy)
        next_ego: dict[int, np.ndarray] = {}
        rewards: dict[int, float] = {}
        dones: dict[int, bool] = {}
        roots_now: dict[int, int] = {}
        for a in env._agents:  # noqa: SLF001
            if a.agent_id not in e_before:
                continue
            roots_now[a.agent_id] = a.root_ancestor_id
            if a.alive:
                next_ego[a.agent_id] = egocentric_obs(env, a, cfg.vision_radius)
                rewards[a.agent_id] = a.energy - e_before[a.agent_id]
                dones[a.agent_id] = False
            else:
                # Mort à ce tick : reward = death_penalty négatif + dones=True
                next_ego[a.agent_id] = last_ego.get(
                    a.agent_id, np.zeros(policy.obs_dim, dtype=np.float32)
                )
                rewards[a.agent_id] = -10.0
                dones[a.agent_id] = True
        # Observe les transitions
        policy.observe_dict(
            prev_obs_ego=last_ego,
            actions=actions,
            rewards=rewards,
            next_obs_ego=next_ego,
            dones=dones,
            agent_root_ids=roots_now,
        )
        # Cull lignées éteintes périodiquement
        if t % 200 == 199:
            alive_roots = {a.root_ancestor_id for a in env._agents if a.alive}  # noqa: SLF001
            policy.registry.cull_dead_lineages(alive_roots)
        # Préparer prochain tick : nouveaux nés ont besoin de leur ego
        for a in env._agents:  # noqa: SLF001
            if a.alive and a.agent_id not in last_ego:
                last_ego[a.agent_id] = egocentric_obs(env, a, cfg.vision_radius)
        last_ego = next_ego
        last_actions = actions
        last_roots = roots_now

    dt = time.time() - t0
    summary = _summary(env, dt, label=f"LineageDQN ({device})")
    summary["n_brains_active"] = len(policy.registry)
    summary["total_brain_steps"] = policy.registry.total_global_steps()
    return summary


def _summary(env, dt: float, label: str) -> dict:
    n_alive = env.n_alive
    n_births = env.n_births_total
    food_eaten_proxy = sum(a.energy for a in env._agents if a.alive) / max(n_alive, 1)  # noqa: SLF001
    lineage_counts = Counter(
        a.root_ancestor_id for a in env._agents if a.alive  # noqa: SLF001
    )
    dom = lineage_counts.most_common(1)
    dom_pct = (100 * dom[0][1] / n_alive) if dom and n_alive > 0 else 0.0
    return {
        "label": label,
        "n_alive": n_alive,
        "n_births": n_births,
        "n_lineages_alive": len(lineage_counts),
        "dominant_lineage_pct": dom_pct,
        "mean_energy_alive": food_eaten_proxy,
        "duration_s": dt,
    }


def main() -> None:
    n_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    print(f"=== V8-B1 bench : ticks={n_ticks} seed={seed} device={device} ===\n")

    print("--- Baseline SmartHeuristic ---")
    baseline = run_baseline(seed, n_ticks)
    for k, v in baseline.items():
        print(f"  {k}: {v}")

    print("\n--- LineageDQN ---")
    rl = run_lineage_dqn(seed, n_ticks, device)
    for k, v in rl.items():
        print(f"  {k}: {v}")

    print("\n--- COMPARE ---")
    print(f"  alive  baseline={baseline['n_alive']:3d}  rl={rl['n_alive']:3d}")
    print(f"  births baseline={baseline['n_births']:3d}  rl={rl['n_births']:3d}")
    print(
        f"  mean_E baseline={baseline['mean_energy_alive']:.1f}  "
        f"rl={rl['mean_energy_alive']:.1f}"
    )
    print(f"  speed  baseline={baseline['duration_s']:.1f}s  rl={rl['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
