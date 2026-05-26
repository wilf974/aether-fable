"""V8-B2.1 multi-seed validation — confirme robustesse "dialectes émergents".

Lance overnight_v8b1.py sur N seeds différents (mode language), agrège
les métriques d'émergence linguistique, calcule mean/std/min/max.

Objectif scientifique : confirmer que le finding "concentration 99.88 %
par lignée + L2 ≈ 3" n'est pas un artefact de seed=42. Si la majorité
des seeds montrent concentration ≥80 % et L2 ≥1.0 → résultat robuste.

Usage :
    python scripts/multiseed_v8b2.py --seeds 5 --ticks 10000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np


def run_one_seed(seed: int, ticks: int, out_dir: str) -> dict[str, Any] | None:
    """Run un seed, retourne les métriques de la JSON résultat (ou None si fail)."""
    script = os.path.join(
        os.path.dirname(__file__), "overnight_v8b1.py",
    )
    seed_out_dir = os.path.join(out_dir, f"seed{seed}")
    cmd = [
        sys.executable, script,
        "--ticks", str(ticks),
        "--seed", str(seed),
        "--device", "cpu",
        "--snap-every", str(max(ticks // 5, 1000)),
        "--regime", "language",
        "--out-dir", seed_out_dir,
    ]
    print(f"\n=== SEED {seed} (ticks={ticks}) ===")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  SEED {seed} FAILED : exit={result.returncode}")
        print(result.stdout[-500:])
        print("STDERR:", result.stderr[-500:])
        return None
    json_path = os.path.join(seed_out_dir, "overnight_v8b1_seed42.json")
    # Le file name est toujours seed42 hardcoded :( on doit l'extraire d'ailleurs
    # Simple : scanner le dossier pour le json
    for fname in os.listdir(seed_out_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(seed_out_dir, fname)
            break
    if not os.path.exists(json_path):
        print(f"  SEED {seed} no JSON output at {json_path}")
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data["_seed"] = seed
    data["_duration_s"] = dt
    print(f"  SEED {seed} OK in {dt:.1f}s")
    return data


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrège les métriques langage + écologie sur N seeds."""
    n = len(seed_results)
    if n == 0:
        return {}

    def collect(path: list[str]) -> list[float]:
        vals = []
        for r in seed_results:
            cur = r
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    cur = None
                    break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return vals

    def stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"n": 0}
        arr = np.array(vals, dtype=np.float64)
        return {
            "n": len(vals),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": [float(v) for v in vals],
        }

    return {
        "n_seeds": n,
        # Écologie
        "n_alive_final": stats(collect(["final_state", "n_alive"])),
        "n_lineages_final": stats(collect(["criterion_3_selection", "n_lineages_final"])),
        "dominant_lineage_pct": stats(collect(["final_state", "top_lineages", 0, "pct"])),
        # KL et héritage
        "kl_inter_lineages": stats(collect(["criterion_2_divergence", "final_kl_mean"])),
        # Langage
        "lang_concentration": stats(collect(["language_metrics_v8b2", "mean_token_lineage_concentration"])),
        "lang_distance_L2": stats(collect(["language_metrics_v8b2", "mean_inter_lineage_distance"])),
        "lang_entropy_ratio": stats(collect(["language_metrics_v8b2", "entropy_ratio"])),
        "lang_total_vocalize": stats(collect(["language_metrics_v8b2", "total_vocalize_count"])),
        "lang_tokens_per_1000": stats(collect(["language_metrics_v8b2", "tokens_per_1000_ticks"])),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ticks", type=int, default=10000)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument(
        "--out-dir", default="results/v8b2_multiseed",
        help="Dossier racine pour les JSON par seed",
    )
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    seed_results: list[dict[str, Any]] = []
    seeds = [args.seed_start + i for i in range(args.seeds)]
    for s in seeds:
        result = run_one_seed(s, args.ticks, args.out_dir)
        if result is not None:
            seed_results.append(result)

    dt_total = time.time() - t0
    agg = aggregate(seed_results)

    # Sauve aggregate
    agg_path = os.path.join(args.out_dir, "aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"V8-B2.1 MULTI-SEED VALIDATION ({args.seeds} seeds × {args.ticks} ticks)")
    print(f"Total time : {dt_total/60:.1f} min")
    print(f"Successful seeds : {agg['n_seeds']}/{args.seeds}")
    print("=" * 60)

    def line(label: str, key: str, fmt: str = "{:.2f}") -> None:
        s = agg.get(key, {})
        if s.get("n", 0) == 0:
            print(f"  {label:30s} : no data")
            return
        print(
            f"  {label:30s} : mean={fmt.format(s['mean'])}  "
            f"std={fmt.format(s['std'])}  "
            f"min={fmt.format(s['min'])}  max={fmt.format(s['max'])}"
        )

    print("\n--- ECOLOGIE ---")
    line("n_alive_final", "n_alive_final", "{:.0f}")
    line("n_lineages_final", "n_lineages_final", "{:.0f}")
    line("dominant_lineage_pct", "dominant_lineage_pct", "{:.1f}%")
    line("kl_inter_lineages", "kl_inter_lineages")
    print("\n--- LANGUAGE EMERGENCE ---")
    line("concentration par lignée", "lang_concentration", "{:.2%}")
    line("distance L2 inter-vocabs", "lang_distance_L2")
    line("entropy ratio", "lang_entropy_ratio", "{:.2%}")
    line("total vocalize", "lang_total_vocalize", "{:.0f}")
    line("tokens / 1000 ticks", "lang_tokens_per_1000", "{:.0f}")

    print(f"\nAggregate JSON saved : {agg_path}")

    # Verdict
    print("\n--- VERDICT ---")
    conc = agg.get("lang_concentration", {})
    l2 = agg.get("lang_distance_L2", {})
    if conc.get("n", 0) > 0 and l2.get("n", 0) > 0:
        c_mean = conc["mean"]
        l_mean = l2["mean"]
        if c_mean >= 0.80 and l_mean >= 1.0:
            print(f"  ROBUST FINDING : concentration {c_mean:.1%} >= 80% ET L2 {l_mean:.2f} >= 1.0")
            print("  Le langage emerge avec dialectes par lignee, sur multiples seeds.")
        elif c_mean >= 0.60:
            print(f"  MODERATE FINDING : concentration {c_mean:.1%} dans [60-80%]")
            print("  Tendance reelle mais variance plus elevee. Plus de seeds recommandes.")
        else:
            print(f"  WEAK FINDING : concentration {c_mean:.1%} < 60%")
            print("  Le finding seed=42 etait peut-etre atypique.")


if __name__ == "__main__":
    main()
