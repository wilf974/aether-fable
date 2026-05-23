"""V5.3 benchmark — cache OFF vs ON sur env tribe (V5.2 family) avec famine.

Question scientifique :
    La prévoyance (cache food) aide-t-elle la survie pendant les famines saisonnières ?

Modes comparés :
- TRIBE_NO_CACHE   : V5.2 family, pas de cache
- TRIBE_WITH_CACHE : V5.2 family + V5.3 cache

L'env est calibré pour avoir une famine d'hiver (winter_factor bas) où
le cache devrait briller.

Usage :
    python scripts/v5_3_benchmark_cache.py --steps 3000 --n-seeds 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from aetherlife.metrics.episode_report import EpisodeStatsTracker
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


SEASON_LABELS = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}


def build_env(mode: str, seed: int, args) -> SeasonalMultiAgentFoodGrid:
    """mode in {tribe_no_cache, tribe_with_cache}."""
    cache_enabled = mode == "tribe_with_cache"
    seasonal = SeasonalConfig(
        season_period=args.season_period,
        spring_lambda_factor=1.5, summer_lambda_factor=1.0,
        autumn_lambda_factor=0.9, winter_lambda_factor=0.25,  # winter dur
    )
    cfg = SeasonalMultiAgentConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        max_energy=200.0, start_energy=100.0,
        metabolism=0.6, food_value=14.0, death_penalty=5.0,
        initial_food_density=0.06, food_respawn_lambda=1.3,
        max_steps=args.steps, seasonal=seasonal,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=130.0, energy_cost=55.0,
            cooldown_ticks=40, max_population=40,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=115.0, build_cost=30.0,
            rest_bonus=3.0, cooldown_ticks=100,
            family_inheritance=True,    # tribe always
        ),
        cache=CacheConfig(
            enabled=cache_enabled,
            deposit_threshold=140.0,
            withdrawal_threshold=35.0,
            max_capacity=60.0,
            deposit_amount=6.0,
            withdrawal_amount=6.0,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    return env


def _lineage_survival_rate(env: SeasonalMultiAgentFoodGrid, n_initial: int) -> float:
    surviving = sum(
        1 for root in range(n_initial)
        if env.has_living_descendant(root)
    )
    return surviving / max(n_initial, 1)


def run_one(mode: str, seed: int, args) -> dict:
    env = build_env(mode, seed, args)
    rng = np.random.default_rng(seed * 7 + 13)
    tracker = EpisodeStatsTracker(n_agents=env.cfg.n_agents, track_seasons=True)
    tracker.reset(env)
    food_eaten = 0

    while env.n_alive > 0 and env.step_count < env.cfg.max_steps:
        actions = {aid: int(rng.integers(0, 4)) for aid in env.alive_agent_ids}
        _, _, _, _, infos = env.step(actions)
        tracker.on_step(env, infos)
        for info in infos.values():
            if info.get("ate"):
                food_eaten += 1

    report = tracker.finalize(env)
    return {
        "mode": mode,
        "seed": seed,
        "n_alive_final": env.n_alive,
        "alive_rate_final": env.n_alive / max(env.cfg.n_agents, 1),
        "births_total": env.n_births_total,
        "nests_built": len(env.nests),
        "nest_visits_total": env.nest_visits_total,
        "rest_energy_gained_total": env.rest_energy_gained_total,
        "cache_deposits_total": env.cache_deposits_total,
        "cache_withdrawals_total": env.cache_withdrawals_total,
        "cache_energy_deposited_total": env.cache_energy_deposited_total,
        "cache_energy_withdrawn_total": env.cache_energy_withdrawn_total,
        "total_cached_food_final": env.total_cached_food,
        "food_eaten_total": food_eaten,
        "mean_lifespan": report.mean_lifespan,
        "lineage_survival_rate": _lineage_survival_rate(env, args.n_agents),
        "mortality_by_season": dict(report.mortality_by_season or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--season-period", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument(
        "--out", type=str, default="logs/v5_3_cache_benchmark.json",
    )
    args = parser.parse_args()

    print(
        f"AetherLife V5.3 benchmark cache OFF vs ON (env tribe)\n"
        f"  env: {args.rows}x{args.cols}  N={args.n_agents}  "
        f"steps={args.steps}  period={args.season_period}\n"
        f"  winter_factor=0.25 (famine hivernale)\n"
        f"  seeds: {list(range(args.n_seeds))}\n"
    )

    modes = ["tribe_no_cache", "tribe_with_cache"]
    all_results: dict[str, list[dict]] = {m: [] for m in modes}
    t0 = time.time()
    for seed in range(args.n_seeds):
        print(f"=== seed {seed} ===")
        for mode in modes:
            label = "NO_CACHE" if mode == "tribe_no_cache" else "+CACHE  "
            print(f"  -> {label} ...", end=" ", flush=True)
            t_a = time.time()
            r = run_one(mode, seed, args)
            dt = time.time() - t_a
            print(
                f"alive={r['n_alive_final']:3d}  "
                f"births={r['births_total']:3d}  "
                f"life={r['mean_lifespan']:5.0f}  "
                f"deposits={r['cache_deposits_total']:4d}  "
                f"withdraws={r['cache_withdrawals_total']:4d}  "
                f"cache_food_final={r['total_cached_food_final']:5.1f}  "
                f"({dt:.1f}s)"
            )
            all_results[mode].append(r)
        print()
    elapsed = time.time() - t0

    def agg(rs: list[dict], key: str) -> tuple[float, float]:
        vals = [r[key] for r in rs]
        if not vals:
            return 0.0, 0.0
        return (
            statistics.mean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        )

    metrics = [
        "alive_rate_final", "births_total", "nests_built",
        "nest_visits_total", "rest_energy_gained_total",
        "cache_deposits_total", "cache_withdrawals_total",
        "cache_energy_deposited_total", "cache_energy_withdrawn_total",
        "total_cached_food_final", "food_eaten_total",
        "mean_lifespan", "lineage_survival_rate",
    ]
    print(f"=== AGREGATION (n_seeds={args.n_seeds}) ===")
    header = f"  {'metric':<32s}"
    for m in modes:
        header += f"  {m:>18s}"
    print(header)
    print("  " + "-" * 32 + ("  " + "-" * 18) * len(modes))
    summary: dict[str, dict] = {}
    for met in metrics:
        row = f"  {met:<32s}"
        sub: dict[str, dict] = {}
        for mode in modes:
            mean, std = agg(all_results[mode], met)
            row += f"  {mean:>11.1f}+/-{std:<5.1f}"
            sub[mode] = {"mean": mean, "std": std}
        print(row)
        summary[met] = sub

    print("\n  mortality_by_season (somme sur seeds) :")
    season_agg: dict[str, dict[int, int]] = {m: {} for m in modes}
    for mode in modes:
        for r in all_results[mode]:
            for s, n in r["mortality_by_season"].items():
                season_agg[mode][s] = season_agg[mode].get(s, 0) + n
    all_seasons = sorted({s for d in season_agg.values() for s in d.keys()})
    sh = f"    {'season':<8s}"
    for m in modes:
        sh += f"  {m:>15s}"
    print(sh)
    for s in all_seasons:
        row = f"    {SEASON_LABELS.get(s, str(s)):<8s}"
        for m in modes:
            row += f"  {season_agg[m].get(s, 0):>15d}"
        print(row)

    print(f"\nTotal time : {elapsed:.1f}s")

    # Verdicts
    print("\n=== VERDICT ===")
    d_life = (
        summary["mean_lifespan"]["tribe_with_cache"]["mean"]
        - summary["mean_lifespan"]["tribe_no_cache"]["mean"]
    )
    d_alive = (
        summary["alive_rate_final"]["tribe_with_cache"]["mean"]
        - summary["alive_rate_final"]["tribe_no_cache"]["mean"]
    )
    d_births = (
        summary["births_total"]["tribe_with_cache"]["mean"]
        - summary["births_total"]["tribe_no_cache"]["mean"]
    )
    d_lineage = (
        summary["lineage_survival_rate"]["tribe_with_cache"]["mean"]
        - summary["lineage_survival_rate"]["tribe_no_cache"]["mean"]
    )
    d_winter_mort = (
        season_agg["tribe_with_cache"].get(3, 0)
        - season_agg["tribe_no_cache"].get(3, 0)
    )
    print(f"  CACHE vs NO_CACHE :")
    print(f"    Dlifespan       = {d_life:+.1f}")
    print(f"    Dalive_rate     = {d_alive:+.3f}")
    print(f"    Dlineage_surv   = {d_lineage:+.3f}")
    print(f"    Dbirths         = {d_births:+.1f}")
    print(f"    Dwinter_deaths  = {d_winter_mort:+d}  (negatif = cache sauve l'hiver)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "runs": all_results,
                "summary": summary,
                "mortality_by_season": season_agg,
            },
            indent=2,
        )
    )
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
