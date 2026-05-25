"""V8-C2.b'' multi-seed ablation — confirme robustesse Δ naissances.

Lance N seeds × 2 runs (témoin + ablation@T) en mode coordination_hard,
agrège les deltas par seed, calcule mean/std.

Si la chute moyenne des naissances est stable et > 10 %, le finding
"communication a une fonction" est solide multi-seed.

Usage :
    python scripts/multiseed_ablation.py --seeds 5 --ticks 10000 \\
        --ablation-tick 5000 --regime coordination_hard
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def _run_one(
    seed: int, ticks: int, ablation_tick: int | None,
    regime: str, out_dir: str,
) -> dict | None:
    """Lance un run (témoin ou ablation) et lit le JSON résultat."""
    script = os.path.join(os.path.dirname(__file__), "overnight_v8b1.py")
    label = f"ablation{ablation_tick}" if ablation_tick else "control"
    seed_dir = os.path.join(out_dir, f"seed{seed}_{label}")
    cmd = [
        sys.executable, script,
        "--ticks", str(ticks),
        "--seed", str(seed),
        "--device", "cpu",
        "--snap-every", str(max(ticks // 4, 1000)),
        "--regime", regime,
        "--out-dir", seed_dir,
    ]
    if ablation_tick is not None:
        cmd += ["--vocalize-disable-after", str(ablation_tick)]
    print(f"  Running seed={seed} {label}...")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    res = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"    FAILED ({dt:.0f}s): {res.stderr[-300:]}")
        return None
    # Find JSON output
    for f in os.listdir(seed_dir):
        if f.endswith(".json"):
            with open(os.path.join(seed_dir, f), encoding="utf-8") as g:
                d = json.load(g)
            print(f"    OK ({dt:.0f}s)")
            return d
    print(f"    No JSON in {seed_dir}")
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ticks", type=int, default=10000)
    p.add_argument("--ablation-tick", type=int, default=5000)
    p.add_argument("--regime", default="coordination_hard")
    p.add_argument(
        "--seed-start", type=int, default=42,
        help="Premier seed (42, 43, ...)",
    )
    p.add_argument(
        "--out-dir", default="results/v8c2b_multiseed_ablation",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    deltas = []
    per_seed = []
    seeds = [args.seed_start + i for i in range(args.seeds)]
    t0 = time.time()
    for s in seeds:
        print(f"\n=== SEED {s} ===")
        ctrl = _run_one(s, args.ticks, None, args.regime, args.out_dir)
        abl = _run_one(
            s, args.ticks, args.ablation_tick, args.regime, args.out_dir,
        )
        if ctrl is None or abl is None:
            print(f"  SKIP seed {s}")
            continue
        c_alive = ctrl["final_state"]["n_alive"]
        a_alive = abl["final_state"]["n_alive"]
        c_births = ctrl["final_state"]["n_births_total"]
        a_births = abl["final_state"]["n_births_total"]
        c_lin = ctrl["criterion_3_selection"]["n_lineages_final"]
        a_lin = abl["criterion_3_selection"]["n_lineages_final"]

        def pct(b, a):
            return (a - b) / max(abs(b), 1) * 100 if b != 0 else 0.0

        delta = {
            "seed": s,
            "control_alive": c_alive, "ablation_alive": a_alive,
            "control_births": c_births, "ablation_births": a_births,
            "control_lineages": c_lin, "ablation_lineages": a_lin,
            "delta_alive_pct": pct(c_alive, a_alive),
            "delta_births_pct": pct(c_births, a_births),
            "delta_lineages_pct": pct(c_lin, a_lin),
        }
        per_seed.append(delta)
        deltas.append(delta["delta_births_pct"])

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"V8-C2.b'' MULTI-SEED ABLATION (seeds={args.seeds}, "
          f"ticks={args.ticks}, ablation@{args.ablation_tick})")
    print(f"Total time : {dt/60:.1f} min")
    print("=" * 60)

    print(f"\n{'Seed':>5} {'C_alv':>5} {'A_alv':>5} {'C_brth':>7} "
          f"{'A_brth':>7} {'Δ_brth':>10}")
    for d in per_seed:
        print(f"{d['seed']:>5} {d['control_alive']:>5} {d['ablation_alive']:>5}"
              f" {d['control_births']:>7} {d['ablation_births']:>7}"
              f" {d['delta_births_pct']:>+9.1f}%")

    if deltas:
        arr = np.array(deltas)
        print(f"\nΔ births (%) : mean={arr.mean():+.2f}, std={arr.std():.2f}, "
              f"min={arr.min():+.2f}, max={arr.max():+.2f}")
        n_negative = sum(1 for d in deltas if d < -5)
        n_strong = sum(1 for d in deltas if d < -15)
        print(f"\nSeeds avec Δ births < -5 %  : {n_negative}/{len(deltas)}")
        print(f"Seeds avec Δ births < -15 % : {n_strong}/{len(deltas)}")

        if arr.mean() < -10 and n_negative >= len(deltas) * 0.6:
            verdict = "communication_fonctionnelle_robuste"
        elif arr.mean() < -5:
            verdict = "communication_fonctionnelle_partielle"
        else:
            verdict = "communication_decorative_ou_seed_dependent"
        print(f"\nVERDICT : {verdict}")

    out_path = os.path.join(args.out_dir, "aggregate.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "seeds": args.seeds, "ticks": args.ticks,
                "ablation_tick": args.ablation_tick, "regime": args.regime,
            },
            "per_seed": per_seed,
            "deltas_births": deltas,
            "delta_births_mean": float(np.mean(deltas)) if deltas else 0,
            "delta_births_std": float(np.std(deltas)) if deltas else 0,
        }, f, indent=2, default=str)
    print(f"\nAggregate saved : {out_path}")


if __name__ == "__main__":
    main()
