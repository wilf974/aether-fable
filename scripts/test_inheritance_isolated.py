"""V8-B1 critère 4 strict : test isolé gen0 vs genN.

Procédure :
    1. Train V8-B1 sur 20k ticks → cerveaux compétents
    2. Sauve les poids du cerveau de la lignée dominante (= "genN")
    3. Reset env, init 5 agents avec cerveau RANDOM (= "gen0")
    4. Reset env, init 5 agents avec cerveau "genN" cloné
    5. Run isolé chaque groupe 2k ticks, mesure lifespan moyen
    6. Si genN > gen0 significativement (>20%), héritage prouvé

C'est la version "double-blind" : on contrôle tout sauf le poids initial.
"""
from __future__ import annotations

import argparse
import copy
import sys

import numpy as np

from aetherlife.agents.lineage_agent import LineageAgent, egocentric_obs
from aetherlife.agents.lineage_brain import BrainConfig
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig, SeasonalMultiAgentConfig, SeasonalMultiAgentFoodGrid,
)


def build_test_env(seed: int, n_agents: int = 5) -> SeasonalMultiAgentFoodGrid:
    """Env permissive identique pour les 2 conditions."""
    cfg = SeasonalMultiAgentConfig(
        rows=20, cols=20, n_agents=n_agents, max_energy=300.0,
        start_energy=160.0, metabolism=0.4, food_value=18.0,
        death_penalty=0.0,
        initial_food_density=0.06, food_respawn_lambda=0.25,
        max_steps=10_000,
        seasonal=SeasonalConfig(season_period=200),
        reproduction=ReproductionConfig(
            enabled=False,  # PAS de repro : on veut juste lifespan
            energy_threshold=200.0, energy_cost=70.0,
            cooldown_ticks=999_999, max_population=n_agents,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=130.0, build_cost=40.0,
            rest_bonus=4.0, cooldown_ticks=100,
        ),
        cache=CacheConfig(enabled=False),
        planting=PlantingConfig(enabled=False),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    return env


def train_dominant_brain(train_ticks: int, seed: int, device: str) -> dict:
    """Phase 1 : train un brain "compétent" via gameplay normal."""
    cfg = SeasonalMultiAgentConfig(
        rows=30, cols=30, n_agents=10, max_energy=300.0, start_energy=180.0,
        metabolism=0.35, food_value=18.0, death_penalty=0.0,
        initial_food_density=0.06, food_respawn_lambda=0.25,
        max_steps=train_ticks + 1000,
        seasonal=SeasonalConfig(season_period=200, winter_lambda_factor=0.6),
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=130.0, energy_cost=70.0,
            cooldown_ticks=100, max_population=30,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=130.0, build_cost=40.0,
            rest_bonus=4.0, cooldown_ticks=100, family_inheritance=True,
        ),
        cache=CacheConfig(enabled=False),
        planting=PlantingConfig(enabled=False),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    brain_cfg = BrainConfig(
        enabled=True, device=device, vision_radius=4,
        hidden_dims=(64, 64), lr=5e-4, batch_size=64,
        buffer_capacity=20_000, min_replay_to_learn=500, train_every=4,
        epsilon_start=0.6, epsilon_end=0.05, epsilon_decay_steps=10_000,
        target_sync_steps=300,
    )
    policy = LineageAgent(env=env, cfg=brain_cfg, n_actions=4, seed=seed)
    obs_stub = {a.agent_id: np.zeros(10) for a in env._agents if a.alive}  # noqa: SLF001
    last_ego = {
        a.agent_id: egocentric_obs(env, a, brain_cfg.vision_radius)
        for a in env._agents  # noqa: SLF001
        if a.alive
    }
    print(f"[Train] Training {train_ticks} ticks...")
    for t in range(1, train_ticks + 1):
        if env.n_alive == 0:
            print(f"[Train] EXTINCTION at t={t}")
            break
        obs_stub = {a.agent_id: np.zeros(10) for a in env._agents if a.alive}  # noqa: SLF001
        actions = policy.act_dict(obs_stub, greedy=False)
        e_before = {
            a.agent_id: a.energy for a in env._agents if a.alive  # noqa: SLF001
        }
        env.step(actions)
        ego_after, rewards, dones, roots = {}, {}, {}, {}
        for a in env._agents:  # noqa: SLF001
            if a.agent_id not in e_before:
                continue
            roots[a.agent_id] = a.root_ancestor_id
            if a.alive:
                ego_after[a.agent_id] = egocentric_obs(env, a, brain_cfg.vision_radius)
                rewards[a.agent_id] = (a.energy - e_before[a.agent_id]) * 0.1
                dones[a.agent_id] = False
            else:
                ego_after[a.agent_id] = last_ego.get(
                    a.agent_id, np.zeros(policy.obs_dim, dtype=np.float32),
                )
                rewards[a.agent_id] = -5.0
                dones[a.agent_id] = True
        policy.observe_dict(
            prev_obs_ego=last_ego, actions=actions, rewards=rewards,
            next_obs_ego=ego_after, dones=dones, agent_root_ids=roots,
        )
        for a in env._agents:  # noqa: SLF001
            if a.alive and a.agent_id not in last_ego:
                last_ego[a.agent_id] = egocentric_obs(env, a, brain_cfg.vision_radius)
        last_ego = ego_after
        if t % 2000 == 0:
            from collections import Counter
            roots_alive = Counter(
                a.root_ancestor_id for a in env._agents if a.alive  # noqa: SLF001
            )
            print(
                f"[Train t={t:5d}] alive={env.n_alive:3d} lineages={len(roots_alive)} "
                f"births={env.n_births_total:3d} steps={policy.registry.total_global_steps()}"
            )

    # Trouver la lignée dominante
    from collections import Counter
    roots_counter = Counter(
        a.root_ancestor_id for a in env._agents if a.alive  # noqa: SLF001
    )
    if not roots_counter:
        print("[Train] FAIL : aucune lignée vivante")
        return {}
    dom_root, dom_n = roots_counter.most_common(1)[0]
    dom_brain = policy.registry.get(dom_root)
    if dom_brain is None:
        print("[Train] FAIL : brain absent")
        return {}
    print(
        f"[Train] Lignée dominante : root={dom_root}, alive={dom_n}, "
        f"global_step={dom_brain.global_step}"
    )
    return {
        "brain": dom_brain,
        "config": brain_cfg,
        "dom_root": dom_root,
    }


def run_isolated_test(
    brain_template, brain_cfg: BrainConfig, n_agents: int, n_ticks: int,
    seed: int, label: str, use_trained: bool,
) -> dict:
    """Spawn `n_agents` agents avec brain (trained ou random) et mesure
    leur lifespan moyen sur `n_ticks` ticks sans reproduction."""
    import torch
    env = build_test_env(seed, n_agents=n_agents)
    policy = LineageAgent(env=env, cfg=brain_cfg, n_actions=4, seed=seed)
    # OVERRIDE : remplacer chaque brain par une copie du brain trained OU random init
    if use_trained:
        for root_id in list(policy.registry.alive_roots()):
            new_brain = type(brain_template).inherit_from(
                parent=brain_template, root_id=root_id,
                mutation_std=0.0, seed=seed + root_id,
            )
            # Force greedy : pas d'exploration pour test "pur"
            new_brain.cfg = type(brain_template.cfg)(
                **{**brain_template.cfg.__dict__,
                   "epsilon_start": 0.05, "epsilon_end": 0.05}
            )
            policy.registry._brains[root_id] = new_brain  # noqa: SLF001
    else:
        # Random init brains (déjà fait par registry init) — juste force epsilon bas
        for brain in policy.registry:
            brain.cfg = type(brain.cfg)(
                **{**brain.cfg.__dict__,
                   "epsilon_start": 0.05, "epsilon_end": 0.05}
            )
    deaths: list[int] = []
    last_seen = set(env.alive_agent_ids)
    for t in range(1, n_ticks + 1):
        if env.n_alive == 0:
            break
        obs_stub = {a.agent_id: np.zeros(10) for a in env._agents if a.alive}  # noqa: SLF001
        actions = policy.act_dict(obs_stub, greedy=True)
        env.step(actions)
        cur_alive = set(env.alive_agent_ids)
        for did in last_seen - cur_alive:
            deaths.append(t)
        last_seen = cur_alive
    lifespans = deaths + [n_ticks] * env.n_alive  # survivants = n_ticks
    return {
        "label": label,
        "use_trained": use_trained,
        "n_agents": n_agents,
        "n_ticks": n_ticks,
        "n_alive_end": env.n_alive,
        "n_deaths": len(deaths),
        "mean_lifespan": float(np.mean(lifespans)) if lifespans else 0.0,
        "median_lifespan": float(np.median(lifespans)) if lifespans else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-ticks", type=int, default=10000)
    p.add_argument("--test-ticks", type=int, default=2000)
    p.add_argument("--n-test-agents", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print("=" * 60)
    print(f"V8-B1 critère 4 : test gen0 vs genN isolé")
    print("=" * 60)

    # Phase 1 : Train
    trained = train_dominant_brain(args.train_ticks, args.seed, args.device)
    if not trained:
        print("Échec entraînement.")
        sys.exit(1)

    print("\n--- Phase 2 : Test isolé ---\n")

    # Test gen0 (random init)
    print(f"\n[gen0] Random brains, {args.n_test_agents} agents, {args.test_ticks} ticks")
    gen0 = run_isolated_test(
        brain_template=trained["brain"], brain_cfg=trained["config"],
        n_agents=args.n_test_agents, n_ticks=args.test_ticks,
        seed=args.seed + 1, label="gen0_random", use_trained=False,
    )
    print(
        f"  alive={gen0['n_alive_end']}/{gen0['n_agents']}  deaths={gen0['n_deaths']}  "
        f"mean_lifespan={gen0['mean_lifespan']:.0f}"
    )

    # Test genN (trained init)
    print(f"\n[genN] Trained brains, {args.n_test_agents} agents, {args.test_ticks} ticks")
    genN = run_isolated_test(
        brain_template=trained["brain"], brain_cfg=trained["config"],
        n_agents=args.n_test_agents, n_ticks=args.test_ticks,
        seed=args.seed + 1, label="genN_trained", use_trained=True,
    )
    print(
        f"  alive={genN['n_alive_end']}/{genN['n_agents']}  deaths={genN['n_deaths']}  "
        f"mean_lifespan={genN['mean_lifespan']:.0f}"
    )

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    delta = genN["mean_lifespan"] - gen0["mean_lifespan"]
    ratio = (genN["mean_lifespan"] / max(gen0["mean_lifespan"], 1)) - 1
    print(f"gen0 mean lifespan : {gen0['mean_lifespan']:.0f}")
    print(f"genN mean lifespan : {genN['mean_lifespan']:.0f}")
    print(f"Delta : {delta:+.0f}  ({ratio:+.1%})")
    if ratio > 0.2:
        print("HÉRITAGE COGNITIF RÉEL CONFIRMÉ (>20% mieux)")
    elif ratio > 0.05:
        print("Héritage faible mais positif")
    else:
        print("PAS d'héritage cognitif clair")


if __name__ == "__main__":
    main()
