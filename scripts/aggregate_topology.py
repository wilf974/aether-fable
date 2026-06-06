"""Agrégation — généralité du portfolio effect par topologie spatiale.

Grille extinction par (k, n_seed_points). Tranche H_spatial (k=1 ext chute quand
n_seed_points augmente -> fragmenter l'espace sauve la monoculture) vs H_type
(k=1 reste fragile a tout n).

Usage:
    python scripts/aggregate_topology.py results/topo_n4_k1/seed* \
        results/topo_n4_k4/seed* results/topo_n8_k1/seed* ...
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from typing import Any


def extract_topo(report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("config", {})
    fs = report.get("final_state", {}) or {}
    coop = report.get("cooperative_v8c3", {}) or {}
    n_alive = fs.get("n_alive", 0) or 0
    return {
        "seed": cfg.get("seed"),
        "k": cfg.get("n_initial_affinities"),
        "n": cfg.get("n_seed_points"),
        "n_alive": n_alive,
        "gather": coop.get("gather_successes_total", 0) or 0,
        "extinct": n_alive == 0,
    }


def summarize_topo(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple, list] = {}
    for r in rows:
        cells.setdefault((r["k"], r["n"]), []).append(r)
    grid: dict[tuple, dict] = {}
    for key, rs in cells.items():
        grid[key] = {
            "n": len(rs),
            "extinction_pct": round(100 * sum(x["extinct"] for x in rs) / len(rs)),
            "alive_mean": round(st.mean(x["n_alive"] for x in rs), 1),
            "gather_mean": round(st.mean(x["gather"] for x in rs), 1),
        }
    return {"grid": grid}


def main() -> None:
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: aggregate_topology.py <dir> [dir ...]")
        sys.exit(1)
    rows = []
    for d in dirs:
        for f in glob.glob(f"{d}/overnight_v8b1_seed*.json"):
            with open(f, encoding="utf-8") as fh:
                rows.append(extract_topo(json.load(fh)))
    s = summarize_topo(rows)
    ns = sorted({k[1] for k in s["grid"]})
    ks = sorted({k[0] for k in s["grid"]})
    print("n_seed_points : " + "  ".join(f"{n:>8}" for n in ns))
    for k in ks:
        cells = [s["grid"].get((k, n)) for n in ns]
        ext = "  ".join(
            f"{c['extinction_pct']:>3}%({c['n']})" if c else "   --   "
            for c in cells)
        print(f"  k={k} extinction : {ext}")
    for k in ks:
        cells = [s["grid"].get((k, n)) for n in ns]
        al = "  ".join(f"{c['alive_mean']:>8}" if c else "   --   "
                       for c in cells)
        print(f"  k={k} alive_moy  : {al}")
    for k in ks:
        cells = [s["grid"].get((k, n)) for n in ns]
        ga = "  ".join(f"{c['gather_mean']:>8}" if c else "   --   "
                       for c in cells)
        print(f"  k={k} gather_moy : {ga}")
    # verdict H_spatial vs H_type (sur k=1)
    k1 = [(n, s["grid"].get((1, n))) for n in ns]
    k1 = [(n, c["extinction_pct"]) for n, c in k1 if c]
    if len(k1) >= 2:
        first, last = k1[0][1], k1[-1][1]
        if last < first - 15:
            verdict = "H_spatial (fragmenter sauve la monoculture)"
        elif last > first - 15:
            verdict = "H_type (k=1 reste fragile a tout n)"
        else:
            verdict = "ambigu"
        print(f"\n--- k=1 extinction {k1[0][0]}->{k1[-1][0]} seeds : "
              f"{first}% -> {last}%  => {verdict} ---")


if __name__ == "__main__":
    main()
