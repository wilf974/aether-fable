"""V5 benchmark — build OFF vs ON, mêmes seeds, config civ, 3000 steps random.

Objectif : valider empiriquement que la construction de nids améliore la
survie / la reproduction sans rendre le monde trivial.

Métriques comparées :
- alive_rate (fin d'épisode + moyen sur la run)
- births (total naissances)
- nests_built (build ON uniquement)
- nest_visits_total, rest_energy_gained_total (build ON uniquement)
- food_eaten (total)
- mean_lifespan (moyenne par agent)
- mortality_by_season (V3 saisonnier)

Usage :
    python scripts/v5_benchmark_build_off_vs_on.py
    python scripts/v5_benchmark_build_off_vs_on.py --steps 3000 --n-seeds 3
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


def build_env(build_on: bool, seed: int, args) -> SeasonalMultiAgentFoodGrid:
    """Construit un env civ avec build on/off (même config sinon).

    Note V5.1-bench : env durci pour ne pas saturer max_population (sinon
    alive_rate identique build OFF/ON et l'effet du nid invisible).
    """
    seasonal = SeasonalConfig(
        season_period=args.season_period,
        spring_lambda_factor=args.spring_factor,
        summer_lambda_factor=args.summer_factor,
        autumn_lambda_factor=args.autumn_factor,
        winter_lambda_factor=args.winter_factor,
    )
    cfg = SeasonalMultiAgentConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        max_energy=args.max_energy, start_energy=args.start_energy,
        metabolism=args.metabolism, food_value=args.food_value,
        death_penalty=5.0,
        initial_food_density=args.density, food_respawn_lambda=args.respawn,
        max_steps=args.steps,
        seasonal=seasonal,
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=args.repro_threshold,
            energy_cost=args.repro_cost,
            cooldown_ticks=args.repro_cooldown, max_population=args.max_pop,
        ),
        build=BuildConfig(
            enabled=build_on,
            energy_threshold=args.build_threshold,
            build_cost=args.build_cost,
            rest_bonus=args.rest_bonus,
            cooldown_ticks=args.build_cooldown,
        ),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    return env


def run_one(build_on: bool, seed: int, args) -> dict:
    """Run un episode complet et retourne les métriques."""
    env = build_env(build_on, seed, args)
    rng = np.random.default_rng(seed * 7 + 13)
    tracker = EpisodeStatsTracker(
        n_agents=env.cfg.n_agents, track_seasons=True,
    )
    tracker.reset(env)

    food_eaten = 0
    while env.n_alive > 0 and env.step_count < env.cfg.max_steps:
        actions = {aid: int(rng.integers(0, 4)) for aid in env.alive_agent_ids}
        _, _, terminated, _, infos = env.step(actions)
        tracker.on_step(env, infos)
        for info in infos.values():
            if info.get("ate"):
                food_eaten += 1

    report = tracker.finalize(env)

    return {
        "build_on": build_on,
        "seed": seed,
        "final_step": env.step_count,
        "n_alive_final": env.n_alive,
        "total_agents_emitted": env.cfg.n_agents + env.n_births_total,
        "alive_rate_final": env.n_alive / max(env.cfg.n_agents, 1),
        "births_total": env.n_births_total,
        "nests_built": len(env.nests),                    # nids existants en fin
        "nest_visits_total": env.nest_visits_total,
        "rest_energy_gained_total": env.rest_energy_gained_total,
        "food_eaten_total": food_eaten,
        "mean_lifespan": report.mean_lifespan,
        "mortality_by_season": dict(report.mortality_by_season or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--season-period", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=3)
    # Env tunables (defaults env durci pour ne pas saturer cap pop)
    parser.add_argument("--max-energy", type=float, default=150.0)
    parser.add_argument("--start-energy", type=float, default=80.0)
    parser.add_argument("--metabolism", type=float, default=0.7)
    parser.add_argument("--food-value", type=float, default=12.0)
    parser.add_argument("--density", type=float, default=0.04)
    parser.add_argument("--respawn", type=float, default=0.8)
    parser.add_argument("--spring-factor", type=float, default=1.5)
    parser.add_argument("--summer-factor", type=float, default=1.0)
    parser.add_argument("--autumn-factor", type=float, default=1.0)
    parser.add_argument("--winter-factor", type=float, default=0.3)
    parser.add_argument("--max-pop", type=int, default=64)
    parser.add_argument("--repro-threshold", type=float, default=100.0)
    parser.add_argument("--repro-cost", type=float, default=50.0)
    parser.add_argument("--repro-cooldown", type=int, default=40)
    parser.add_argument("--build-threshold", type=float, default=100.0)
    parser.add_argument("--build-cost", type=float, default=30.0)
    parser.add_argument("--rest-bonus", type=float, default=3.0)
    parser.add_argument("--build-cooldown", type=int, default=80)
    parser.add_argument(
        "--out", type=str, default="logs/v5_benchmark_results.json",
        help="Fichier JSON de sortie pour les résultats détaillés",
    )
    args = parser.parse_args()

    print(
        f"AetherLife V5 benchmark build OFF vs ON\n"
        f"  env: {args.rows}x{args.cols}  N={args.n_agents}  "
        f"steps={args.steps}  season_period={args.season_period}\n"
        f"  seeds: {list(range(args.n_seeds))}\n"
    )

    all_results = {"off": [], "on": []}
    t0 = time.time()
    for seed in range(args.n_seeds):
        print(f"=== seed {seed} ===")
        for build_on, key in [(False, "off"), (True, "on")]:
            print(f"  -> build={'ON ' if build_on else 'OFF'} ...", end=" ", flush=True)
            t_a = time.time()
            r = run_one(build_on, seed, args)
            dt = time.time() - t_a
            print(
                f"alive={r['n_alive_final']:3d}  "
                f"births={r['births_total']:3d}  "
                f"nests_now={r['nests_built']:3d}  "
                f"visits={r['nest_visits_total']:4d}  "
                f"rest_gain={r['rest_energy_gained_total']:7.1f}  "
                f"food={r['food_eaten_total']:5d}  "
                f"life={r['mean_lifespan']:5.0f}  "
                f"({dt:.1f}s)"
            )
            all_results[key].append(r)
        print()

    elapsed = time.time() - t0

    # ─── Agrégation ──────────────────────────────────────────────────────
    def agg(rs: list[dict], key: str) -> tuple[float, float]:
        vals = [r[key] for r in rs]
        if not vals:
            return 0.0, 0.0
        return statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0

    metrics = [
        "alive_rate_final", "births_total", "nests_built",
        "nest_visits_total", "rest_energy_gained_total",
        "food_eaten_total", "mean_lifespan",
    ]
    print(f"=== AGRÉGATION (n_seeds={args.n_seeds}) ===\n")
    print(
        f"  {'metric':<30s}  {'build OFF':>16s}  {'build ON':>16s}  {'delta':>14s}"
    )
    print(f"  {'-'*30}  {'-'*16}  {'-'*16}  {'-'*14}")
    summary = {}
    for m in metrics:
        off_m, off_s = agg(all_results["off"], m)
        on_m, on_s = agg(all_results["on"], m)
        delta = on_m - off_m
        pct = (delta / off_m * 100) if abs(off_m) > 1e-9 else float("inf")
        pct_str = f"{pct:+6.1f}%" if pct != float("inf") else "  inf"
        print(
            f"  {m:<30s}  {off_m:>9.1f}±{off_s:<5.1f}  "
            f"{on_m:>9.1f}±{on_s:<5.1f}  {delta:>+8.1f} {pct_str}"
        )
        summary[m] = {
            "off_mean": off_m, "off_std": off_s,
            "on_mean": on_m, "on_std": on_s,
            "delta_mean": delta, "delta_pct": pct,
        }

    # Mortality by season aggregated
    print("\n  mortality_by_season (off / on, somme sur seeds) :")
    off_mort = {}
    on_mort = {}
    for r in all_results["off"]:
        for s, n in r["mortality_by_season"].items():
            off_mort[s] = off_mort.get(s, 0) + n
    for r in all_results["on"]:
        for s, n in r["mortality_by_season"].items():
            on_mort[s] = on_mort.get(s, 0) + n
    for s in sorted(set(list(off_mort.keys()) + list(on_mort.keys()))):
        label = SEASON_LABELS.get(s, str(s))
        off_n = off_mort.get(s, 0)
        on_n = on_mort.get(s, 0)
        delta = on_n - off_n
        print(f"    {label:8s}  off={off_n:4d}  on={on_n:4d}  delta={delta:+4d}")

    print(f"\nTotal time : {elapsed:.1f}s")

    # ─── Verdict ─────────────────────────────────────────────────────────
    delta_alive = summary["alive_rate_final"]["delta_mean"]
    delta_life = summary["mean_lifespan"]["delta_mean"]
    delta_births = summary["births_total"]["delta_mean"]
    print("\n=== VERDICT ===")
    if delta_alive > 0.05 and delta_life > 0:
        print("  build ON ameliore survie ET lifespan : territoire fonctionnel")
    elif delta_alive > 0:
        print("  build ON ameliore alive_rate mais pas lifespan : effet marginal")
    elif delta_alive < -0.05:
        print("  build ON DEGRADE la survie : reequilibrage necessaire")
    else:
        print("  build ON ~= build OFF : probablement env trop genereux")
    print(f"  delta alive_rate : {delta_alive:+.1%}")
    print(f"  delta lifespan   : {delta_life:+.1f}")
    print(f"  delta births     : {delta_births:+.1f}")

    # ─── Save JSON ───────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "runs": all_results,
                "summary": summary,
                "mortality_by_season": {"off": off_mort, "on": on_mort},
            },
            indent=2,
        )
    )
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
