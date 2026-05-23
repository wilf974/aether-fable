"""Diagnostic longue durée V7 — 10000 ticks, traits + lignées + dominance.

Mesure avant calibration : laisse la sélection naturelle s'exprimer sur des
temps géologiques (10k ticks = ~50 saisons en mode garden) pour observer :
    1. Convergence des traits (build_bias monte-t-il ? plant_bias ?)
    2. Dominance des lignées (un root_ancestor_id devient-il majoritaire ?)
    3. Stabilité population / extinctions partielles
    4. Mortalité par génération (proxy de "pression sélective")
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

import numpy as np

from aetherlife.agents.smart_heuristic import SmartHeuristicAgent
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)
from aetherlife.world.traits import TraitsConfig


def build_env(
    seed: int, *, max_pop: int = 16, cooldown: int = 250,
    metabolism: float = 0.4, winter_factor: float = 0.6,
) -> SeasonalMultiAgentFoodGrid:
    cfg = SeasonalMultiAgentConfig(
        rows=40, cols=40, n_agents=10, max_energy=300.0, start_energy=140.0,
        metabolism=metabolism, food_value=18.0, death_penalty=0.0,
        initial_food_density=0.04, food_respawn_lambda=0.15, max_steps=12000,
        seasonal=SeasonalConfig(
            season_period=200, spring_lambda_factor=1.4,
            summer_lambda_factor=1.0, autumn_lambda_factor=0.8,
            winter_lambda_factor=winter_factor,
        ),
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=130.0, energy_cost=70.0,
            cooldown_ticks=cooldown, max_population=max_pop,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=130.0, build_cost=40.0,
            rest_bonus=4.0, cooldown_ticks=100, family_inheritance=True,
        ),
        cache=CacheConfig(
            enabled=True, deposit_threshold=170.0,
            withdrawal_threshold=70.0, max_capacity=100.0,
            deposit_amount=5.0, withdrawal_amount=5.0,
        ),
        planting=PlantingConfig(
            enabled=True, energy_threshold=110.0, energy_cost=15.0,
            growth_ticks=80, cooldown_ticks=40,
            seeds_required=1, seeds_per_food_eaten=1, initial_seeds=2,
        ),
        traits=TraitsConfig(enabled=True, mutation_std=0.08, initial_std=0.15),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    return env


def lineage_dominance(env) -> tuple[Counter, int, int]:
    """Compte par root_ancestor_id chez les vivants. Renvoie (counter, dom_root, dom_n)."""
    alive_roots = Counter(
        a.root_ancestor_id for a in env._agents if a.alive  # noqa: SLF001
    )
    if not alive_roots:
        return Counter(), -1, 0
    dom_root, dom_n = alive_roots.most_common(1)[0]
    return alive_roots, dom_root, dom_n


def lineage_traits(env, root_id: int) -> np.ndarray | None:
    """Moyenne des traits des agents vivants d'une lignée."""
    arrs = [
        a.traits.as_array() for a in env._agents  # noqa: SLF001
        if a.alive and a.root_ancestor_id == root_id and a.traits is not None
    ]
    if not arrs:
        return None
    return np.array(arrs).mean(axis=0)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    n_ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    max_pop = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    cooldown = int(sys.argv[4]) if len(sys.argv) > 4 else 250
    metabolism = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
    winter_factor = float(sys.argv[6]) if len(sys.argv) > 6 else 0.6
    snap_every = max(1, n_ticks // 20)

    env = build_env(
        seed, max_pop=max_pop, cooldown=cooldown,
        metabolism=metabolism, winter_factor=winter_factor,
    )
    print(
        f"(max_pop={max_pop}, cooldown={cooldown}, "
        f"metabolism={metabolism}, winter={winter_factor})"
    )
    policy = SmartHeuristicAgent(env, seed=seed)
    obs = {a.agent_id: env._observation_for(a.agent_id) for a in env._agents}  # noqa: SLF001

    init_dist = env.trait_distribution
    print(f"=== V7 longrun diagnostic - seed={seed}  ticks={n_ticks} ===")
    print(f"INIT  n={init_dist.n}")
    print(
        f"      mean build={init_dist.mean[0]:.3f} plant={init_dist.mean[1]:.3f} "
        f"cache={init_dist.mean[2]:.3f} explore={init_dist.mean[3]:.3f}"
    )
    print()

    # Historique pour analyse trajectoire
    hist_t: list[int] = []
    hist_mean: list[np.ndarray] = []
    hist_alive: list[int] = []
    hist_births: list[int] = []
    hist_deaths_by_gen: dict[int, int] = defaultdict(int)
    last_alive_set: set[int] = {a.agent_id for a in env._agents if a.alive}  # noqa: SLF001

    print(
        f"{'tick':>5} {'alive':>5} {'births':>6} {'deaths':>6} {'gen_max':>7} "
        f"{'dom_n':>5} {'build':>6} {'plant':>6} {'cache':>6} {'explor':>6}"
    )

    for t in range(1, n_ticks + 1):
        if env.n_alive == 0:
            print(f"[t={t}] EXTINCTION — arrêt du run")
            break
        actions = policy.act_dict(obs, greedy=True)
        obs, _, _, _, _ = env.step(actions)

        # Détection morts (set difference)
        cur_alive_set = {a.agent_id for a in env._agents if a.alive}  # noqa: SLF001
        dead_ids = last_alive_set - cur_alive_set
        for did in dead_ids:
            try:
                gen = env.agent_state(did).generation
                hist_deaths_by_gen[gen] += 1
            except KeyError:
                pass
        last_alive_set = cur_alive_set

        if t % snap_every == 0 or t == n_ticks:
            d = env.trait_distribution
            _, _, dom_n = lineage_dominance(env)
            gen_max = max(
                (a.generation for a in env._agents if a.alive),  # noqa: SLF001
                default=0,
            )
            n_deaths_cum = sum(hist_deaths_by_gen.values())
            print(
                f"{t:>5} {env.n_alive:>5} {env.n_births_total:>6} "
                f"{n_deaths_cum:>6} {gen_max:>7} {dom_n:>5} "
                f"{d.mean[0]:>6.3f} {d.mean[1]:>6.3f} "
                f"{d.mean[2]:>6.3f} {d.mean[3]:>6.3f}"
            )
            hist_t.append(t)
            hist_mean.append(d.mean.copy())
            hist_alive.append(env.n_alive)
            hist_births.append(env.n_births_total)

    print()
    print("---DOMINANCE LIGNÉES---")
    roots_counter, dom_root, dom_n = lineage_dominance(env)
    n_alive = env.n_alive
    if n_alive > 0:
        print(f"Lignées vivantes : {len(roots_counter)} root_ancestors")
        for root, count in roots_counter.most_common(5):
            pct = 100 * count / n_alive
            traits_mean = lineage_traits(env, root)
            traits_str = (
                f"build={traits_mean[0]:.2f} plant={traits_mean[1]:.2f} "
                f"cache={traits_mean[2]:.2f} explore={traits_mean[3]:.2f}"
                if traits_mean is not None else "(no traits)"
            )
            print(f"  root={root:3d}  alive={count:2d} ({pct:4.1f}%)  {traits_str}")
        # Dominance ratio
        dominance_ratio = dom_n / n_alive
        print(f"\nDominance ratio (top lignée / total): {dominance_ratio:.2%}")
        if dominance_ratio >= 0.6:
            print("  => DOMINANCE FORTE : une lignée occupe ≥60% de la population")
        elif dominance_ratio >= 0.4:
            print("  => Dominance modérée")
        else:
            print("  => Diversité préservée (pas de famille dominante)")

    print()
    print("---MORTALITÉ PAR GÉNÉRATION---")
    if hist_deaths_by_gen:
        for gen in sorted(hist_deaths_by_gen):
            print(f"  gen={gen:2d}  morts={hist_deaths_by_gen[gen]:3d}")

    print()
    print("---TRAJECTOIRE TRAITS (delta INIT -> FINAL)---")
    if hist_mean:
        delta = hist_mean[-1] - init_dist.mean
        labels = ["build", "plant", "cache", "explore"]
        for i, lab in enumerate(labels):
            sign = "+" if delta[i] >= 0 else ""
            print(
                f"  {lab:7s}  {init_dist.mean[i]:.3f} → {hist_mean[-1][i]:.3f}  "
                f"({sign}{delta[i]:.3f})"
            )

    print()
    print("---PHASES (5 quintiles)---")
    if len(hist_mean) >= 5:
        n_phases = 5
        step = max(1, len(hist_mean) // n_phases)
        for i in range(0, len(hist_mean), step):
            m = hist_mean[i]
            print(
                f"  t={hist_t[i]:>5d}  alive={hist_alive[i]:>2d}  "
                f"births={hist_births[i]:>3d}  "
                f"build={m[0]:.2f} plant={m[1]:.2f} cache={m[2]:.2f} explore={m[3]:.2f}"
            )


if __name__ == "__main__":
    main()
