"""V5.2 benchmark — 3-way OFF / OWN / FAMILY, mêmes seeds, env mid-range.

Question scientifique :
    Le territoire devient-il un avantage de lignée ?

Modes comparés :
- OFF    : pas de construction (V2/V3/V4 baseline)
- OWN    : V5.0 — rest_bonus owner-only, nid disparaît à la mort
- FAMILY : V5.2 — rest_bonus à toute la lignée (root_ancestor partagé),
           nid persiste tant qu'un descendant vit

Usage :
    python scripts/v5_2_benchmark_family.py --steps 3000 --n-seeds 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from aetherlife.metrics.episode_report import EpisodeStatsTracker
from aetherlife.world.construction import BuildConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


SEASON_LABELS = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}


def build_env(mode: str, seed: int, args) -> SeasonalMultiAgentFoodGrid:
    """mode in {off, own, family}."""
    build_enabled = mode != "off"
    family = mode == "family"
    seasonal = SeasonalConfig(
        season_period=args.season_period,
        spring_lambda_factor=1.5, summer_lambda_factor=1.0,
        autumn_lambda_factor=0.9, winter_lambda_factor=0.4,
    )
    cfg = SeasonalMultiAgentConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        max_energy=200.0, start_energy=100.0,
        metabolism=0.6, food_value=14.0, death_penalty=5.0,
        initial_food_density=0.06, food_respawn_lambda=1.4,
        max_steps=args.steps, seasonal=seasonal,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=130.0, energy_cost=55.0,
            cooldown_ticks=40, max_population=40,
        ),
        build=BuildConfig(
            enabled=build_enabled,
            energy_threshold=115.0, build_cost=30.0,
            rest_bonus=3.0, cooldown_ticks=100,
            family_inheritance=family,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    return env


def _lineage_survival_rate(env: SeasonalMultiAgentFoodGrid, n_initial: int) -> float:
    """% de lignées initiales (root_id < n_initial) avec au moins 1 descendant vivant."""
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
        "family_nest_visits_total": env.family_nest_visits_total,
        "rest_energy_gained_total": env.rest_energy_gained_total,
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
        "--modes", type=str, nargs="+", default=["off", "own", "family"],
        choices=["off", "own", "family"],
    )
    parser.add_argument(
        "--out", type=str, default="logs/v5_2_family_benchmark.json",
    )
    args = parser.parse_args()

    print(
        f"AetherLife V5.2 benchmark family-vs-own-vs-off\n"
        f"  env: {args.rows}x{args.cols}  N={args.n_agents}  "
        f"steps={args.steps}  period={args.season_period}\n"
        f"  modes: {args.modes}  seeds: {list(range(args.n_seeds))}\n"
    )

    all_results: dict[str, list[dict]] = {m: [] for m in args.modes}
    t0 = time.time()
    for seed in range(args.n_seeds):
        print(f"=== seed {seed} ===")
        for mode in args.modes:
            label = {"off": "OFF   ", "own": "OWN   ", "family": "FAMILY"}[mode]
            print(f"  -> {label} ...", end=" ", flush=True)
            t_a = time.time()
            r = run_one(mode, seed, args)
            dt = time.time() - t_a
            print(
                f"alive={r['n_alive_final']:3d}  "
                f"births={r['births_total']:3d}  "
                f"nests={r['nests_built']:3d}  "
                f"visits={r['nest_visits_total']:4d}  "
                f"lineage={r['lineage_survival_rate']:5.1%}  "
                f"life={r['mean_lifespan']:5.0f}  ({dt:.1f}s)"
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
        "food_eaten_total", "mean_lifespan", "lineage_survival_rate",
    ]
    print(f"=== AGREGATION (n_seeds={args.n_seeds}) ===")
    header = f"  {'metric':<28s}"
    for m in args.modes:
        header += f"  {m:>16s}"
    print(header)
    print("  " + "-" * 28 + ("  " + "-" * 16) * len(args.modes))
    summary: dict[str, dict] = {}
    for met in metrics:
        row = f"  {met:<28s}"
        sub: dict[str, dict] = {}
        for mode in args.modes:
            mean, std = agg(all_results[mode], met)
            row += f"  {mean:>9.1f}+/-{std:<5.1f}"
            sub[mode] = {"mean": mean, "std": std}
        print(row)
        summary[met] = sub

    # Mortality by season
    print("\n  mortality_by_season (somme sur seeds) :")
    season_agg: dict[str, dict[int, int]] = {m: {} for m in args.modes}
    for mode in args.modes:
        for r in all_results[mode]:
            for s, n in r["mortality_by_season"].items():
                season_agg[mode][s] = season_agg[mode].get(s, 0) + n
    all_seasons = sorted({s for d in season_agg.values() for s in d.keys()})
    season_header = f"    {'season':<8s}"
    for m in args.modes:
        season_header += f"  {m:>6s}"
    print(season_header)
    for s in all_seasons:
        row = f"    {SEASON_LABELS.get(s, str(s)):<8s}"
        for m in args.modes:
            row += f"  {season_agg[m].get(s, 0):>6d}"
        print(row)

    print(f"\nTotal time : {elapsed:.1f}s")

    # Verdicts
    print("\n=== VERDICT ===")
    if "own" in args.modes and "off" in args.modes:
        d_life = summary["mean_lifespan"]["own"]["mean"] - summary["mean_lifespan"]["off"]["mean"]
        d_alive = summary["alive_rate_final"]["own"]["mean"] - summary["alive_rate_final"]["off"]["mean"]
        print(f"  OWN vs OFF      : Dlifespan={d_life:+6.1f}  Dalive_rate={d_alive:+.3f}")
    if "family" in args.modes and "own" in args.modes:
        d_life = summary["mean_lifespan"]["family"]["mean"] - summary["mean_lifespan"]["own"]["mean"]
        d_alive = summary["alive_rate_final"]["family"]["mean"] - summary["alive_rate_final"]["own"]["mean"]
        d_lin = summary["lineage_survival_rate"]["family"]["mean"] - summary["lineage_survival_rate"]["own"]["mean"]
        d_visits = (
            summary["nest_visits_total"]["family"]["mean"]
            - summary["nest_visits_total"]["own"]["mean"]
        )
        d_births = (
            summary["births_total"]["family"]["mean"]
            - summary["births_total"]["own"]["mean"]
        )
        print(f"  FAMILY vs OWN   : Dlifespan={d_life:+6.1f}  Dalive_rate={d_alive:+.3f}")
        print(f"                    Dlineage_survival={d_lin:+.3f}  Dvisits={d_visits:+.0f}")
        print(f"                    Dbirths={d_births:+.1f}")
    if "family" in args.modes and "off" in args.modes:
        d_life = summary["mean_lifespan"]["family"]["mean"] - summary["mean_lifespan"]["off"]["mean"]
        d_alive = summary["alive_rate_final"]["family"]["mean"] - summary["alive_rate_final"]["off"]["mean"]
        print(f"  FAMILY vs OFF   : Dlifespan={d_life:+6.1f}  Dalive_rate={d_alive:+.3f}")

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
