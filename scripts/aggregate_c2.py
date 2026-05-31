"""Agregation C2 — diversite d'affinite -> mobilite (design apparie).

Lit les reports overnight de results/c2_aff{k}/seed{s}/ et produit :
- une table appariee par seed (mobility k1/k2/k4 + delta intra-seed)
- les moyennes par condition + garde-fous survie
- le test du signe apparie (nb de seeds ou mobility_k1 > mobility_k4)

Usage:
    python scripts/aggregate_c2.py results/c2_aff1/seed* results/c2_aff2/seed* \\
        results/c2_aff4/seed*
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from typing import Any


def extract_c2(report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("config", {})
    sm = report.get("spatial_mobility_v8c3", {}) or {}
    fs = report.get("final_state", {}) or {}
    coop = report.get("cooperative_v8c3", {}) or {}
    aff_dist = fs.get("affinity_distribution", {}) or {}
    counts = [int(v) for v in aff_dist.values()]
    total = sum(counts)
    aff_conc = (max(counts) / total) if total > 0 else 0.0
    n_alive = fs.get("n_alive", 0) or 0
    return {
        "seed": cfg.get("seed"),
        "k": cfg.get("n_initial_affinities"),
        "mobility_score": sm.get("corr_occupation_start_end"),
        "village_basin": sm.get("village_basin"),
        "n_alive": n_alive,
        "gather_successes": coop.get("gather_successes_total", 0) or 0,
        "extinction": n_alive == 0,
        "aff_conc_final": round(aff_conc, 3),
    }


def summarize_c2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[Any, dict[int, dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["k"]] = r
    paired: dict[Any, dict] = {}
    n_k1_gt_k4 = 0
    n_paired = 0
    for seed, byk in by_seed.items():
        if 1 in byk and 4 in byk:
            m1 = byk[1]["mobility_score"]
            m4 = byk[4]["mobility_score"]
            if m1 is not None and m4 is not None:
                n_paired += 1
                delta = m1 - m4
                paired[seed] = {"mobility_k1": m1, "mobility_k4": m4,
                                "delta_k1_k4": delta}
                if m1 > m4:
                    n_k1_gt_k4 += 1
    by_cond: dict[int, dict] = {}
    for k in (1, 2, 4):
        ms = [r["mobility_score"] for r in rows
              if r["k"] == k and r["mobility_score"] is not None]
        vb = [r["village_basin"] for r in rows
              if r["k"] == k and r["village_basin"] is not None]
        alive = [r["n_alive"] for r in rows if r["k"] == k]
        gather = [r["gather_successes"] for r in rows if r["k"] == k]
        ext = [r["extinction"] for r in rows if r["k"] == k]
        if ms:
            by_cond[k] = {
                "mobility_mean": round(st.mean(ms), 3),
                "village_pct": round(100 * sum(vb) / len(vb)) if vb else None,
                "alive_mean": round(st.mean(alive), 1) if alive else None,
                "gather_mean": round(st.mean(gather), 1) if gather else None,
                "extinction_pct": round(100 * sum(ext) / len(ext)) if ext else None,
                "n": len(ms),
            }
    return {
        "paired": paired, "by_cond": by_cond,
        "n_paired": n_paired, "n_seeds_k1_gt_k4": n_k1_gt_k4,
    }


def main() -> None:
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: aggregate_c2.py <clip_dir> [clip_dir ...]")
        sys.exit(1)
    rows = []
    for d in dirs:
        for f in glob.glob(f"{d}/overnight_v8b1_seed*.json"):
            with open(f, encoding="utf-8") as fh:
                rows.append(extract_c2(json.load(fh)))
    s = summarize_c2(rows)
    print(f"{'seed':>5} {'mob_k1':>7} {'mob_k4':>7} {'delta(k1-k4)':>12}")
    for seed, p in sorted(s["paired"].items()):
        print(f"{str(seed):>5} {p['mobility_k1']:>7.3f} "
              f"{p['mobility_k4']:>7.3f} {p['delta_k1_k4']:>+12.3f}")
    print(f"\n--- {s['n_seeds_k1_gt_k4']}/{s['n_paired']} seeds : "
          f"mobility_k1 > mobility_k4 (test du signe apparie) ---")
    print(f"{'k':>3} {'mob':>6} {'village%':>8} {'alive':>6} "
          f"{'gather':>7} {'ext%':>5}")
    for k in (1, 2, 4):
        c = s["by_cond"].get(k)
        if c:
            print(f"{k:>3} {c['mobility_mean']:>6.3f} "
                  f"{str(c['village_pct']):>8} {str(c['alive_mean']):>6} "
                  f"{str(c['gather_mean']):>7} {str(c['extinction_pct']):>5}")


if __name__ == "__main__":
    main()
