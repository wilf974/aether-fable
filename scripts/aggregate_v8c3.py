"""V8-C3 — Aggregation multi-seed pour curriculum C3a''-soft.

Lit N rapports JSON et produit :
  - Tableau seed par seed des 4 métriques
  - Stats agrégées (mean, std, min, max)
  - Verdict probabiliste par seed (3 patterns coop + diagnostic neutre)
  - Verdict global multi-seed selon critères de succès :
      * >= 2/4 patterns positifs sur >= 2/3 seeds
      * OU clustering trend > +1.0 sur >= 2/3 seeds

Sortie : results/v8c3a2soft_aggregate.json + impression console.

Usage:
    python scripts/aggregate_v8c3.py \\
        --runs results/v8c3a2soft_seed42 \\
               results/v8c3a2soft_seed123 \\
               results/v8c3a2soft_seed7 \\
        --out results/v8c3a2soft_aggregate.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _load_report(run_dir: str) -> dict[str, Any]:
    """Charge le JSON principal d'un run (overnight_v8b1_seed*.json)."""
    for fname in os.listdir(run_dir):
        if fname.startswith("overnight_v8b1_seed") and fname.endswith(".json"):
            with open(os.path.join(run_dir, fname), encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No overnight_v8b1_seed*.json in {run_dir}")


def _extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Extrait les métriques V8-C3 d'un rapport."""
    coop = report.get("cooperative_v8c3", {}) or {}
    m = report.get("cooperative_metrics_v8c3", {}) or {}
    fs = report.get("final_state", {}) or {}
    cl = m.get("clustering_pre_success", {}) or {}
    dl = m.get("vocalize_to_gather_delay", {}) or {}
    tk = m.get("token_entropy_pre_success", {}) or {}
    ch = m.get("success_chains", {}) or {}
    n_succ = coop.get("gather_successes_total", 0) or 0
    n_fail = coop.get("gather_failures_total", 0) or 0
    n_cascade = ch.get("n_cascade_successes", 0) or 0
    return {
        "seed": report.get("config", {}).get("seed"),
        "n_alive_final": fs.get("n_alive"),
        "n_births_total": fs.get("n_births_total"),
        "n_deaths": fs.get("n_deaths"),
        "n_lineages_final": (
            report.get("criterion_3_selection", {}) or {}
        ).get("n_lineages_final"),
        "top_lineage_pct": (
            (fs.get("top_lineages") or [{}])[0].get("pct") or 0.0
        ),
        # Coop counts
        "gather_successes": n_succ,
        "gather_failures": n_fail,
        "success_rate": n_succ / max(n_succ + n_fail, 1),
        # Métrique 1
        "clustering_mean": cl.get("mean_neighbors_r3"),
        "clustering_median": cl.get("median_neighbors_r3"),
        "clustering_trend": cl.get("trend_q4_minus_q1"),
        # Métrique 2
        "delay_coverage": dl.get("coverage"),
        "delay_mean": dl.get("mean_min_delay"),
        "delay_trend": dl.get("trend_q4_minus_q1"),
        # Métrique 3
        "token_dominant": tk.get("dominant_token"),
        "token_dominant_share": tk.get("dominant_share"),
        "token_entropy": tk.get("entropy"),
        # Métrique 4
        "n_chains": ch.get("n_chains"),
        "max_chain_len": ch.get("max_chain_len"),
        "n_cascade_successes": n_cascade,
        "cascade_ratio": n_cascade / max(n_succ, 1),
    }


def _verdict_per_seed(m: dict[str, Any]) -> dict[str, bool]:
    """Évalue les 3 seuils par seed."""
    n_succ = m.get("gather_successes", 0) or 0
    cl_trend = m.get("clustering_trend") or 0.0
    delay_trend = m.get("delay_trend") or 0.0
    dom_share = m.get("token_dominant_share") or 0.0
    cascade_ratio = m.get("cascade_ratio") or 0.0
    return {
        "apprenable": (n_succ >= 50) and (cl_trend > 0.0),
        "protocol_emergent": (
            n_succ >= 30 and dom_share > 0.5 and delay_trend < 0.0
        ),
        "cascade_attractor": (
            n_succ >= 30 and cascade_ratio > 0.2
        ),
        "no_extinction": (m.get("n_alive_final") or 0) > 0,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def aggregate(run_dirs: list[str]) -> dict[str, Any]:
    seeds_data = []
    for d in run_dirs:
        try:
            r = _load_report(d)
            m = _extract_metrics(r)
            m["verdict_per_seed"] = _verdict_per_seed(m)
            m["run_dir"] = d
            seeds_data.append(m)
        except FileNotFoundError:
            seeds_data.append({"run_dir": d, "error": "no report found"})

    valid = [s for s in seeds_data if "error" not in s]
    n_valid = len(valid)
    if n_valid == 0:
        return {"seeds": seeds_data, "n_valid": 0, "verdict_global": "no_data"}

    # Stats agrégées
    def _agg(key: str) -> dict[str, float]:
        vals = [s[key] for s in valid if s.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        mean, std = _mean_std(vals)
        return {
            "mean": mean, "std": std, "min": min(vals), "max": max(vals),
        }

    agg = {
        "n_alive_final": _agg("n_alive_final"),
        "n_births_total": _agg("n_births_total"),
        "gather_successes": _agg("gather_successes"),
        "success_rate": _agg("success_rate"),
        "clustering_mean": _agg("clustering_mean"),
        "clustering_trend": _agg("clustering_trend"),
        "delay_coverage": _agg("delay_coverage"),
        "delay_trend": _agg("delay_trend"),
        "token_dominant_share": _agg("token_dominant_share"),
        "token_entropy": _agg("token_entropy"),
        "cascade_ratio": _agg("cascade_ratio"),
    }

    # Verdict global
    n_apprenable = sum(
        s["verdict_per_seed"]["apprenable"] for s in valid
    )
    n_protocol = sum(
        s["verdict_per_seed"]["protocol_emergent"] for s in valid
    )
    n_cascade = sum(
        s["verdict_per_seed"]["cascade_attractor"] for s in valid
    )
    n_no_ext = sum(s["verdict_per_seed"]["no_extinction"] for s in valid)
    n_clustering_strong = sum(
        1 for s in valid
        if (s.get("clustering_trend") or 0.0) > 1.0
        and (s.get("gather_successes") or 0) >= 50
    )

    # Critères de succès
    crit_2_4_patterns = sum([
        n_apprenable >= 2, n_protocol >= 2, n_cascade >= 2,
    ])  # combien de patterns ont >=2 seeds qui le déclenchent
    crit_clustering_strong_majority = (n_clustering_strong >= 2)

    # Verdict extinction : strict seulement si >= 30 % des seeds éteints.
    # Une extinction isolée (1/10) est de la variance reproductrice
    # initiale et ne disqualifie pas le régime.
    n_extinct = n_valid - n_no_ext
    extinction_ratio = n_extinct / max(n_valid, 1)
    if extinction_ratio >= 0.30:
        verdict = "extinction_dominant_design_invalid"
    elif crit_2_4_patterns >= 2 or crit_clustering_strong_majority:
        verdict = "C3b_unlocked_proto_coordination_emerging"
    elif n_apprenable >= 2:
        verdict = "C3a_validated_apprenable_only"
    elif n_no_ext == n_valid and agg["gather_successes"]["mean"] >= 50:
        verdict = "C3a_mechanic_active_no_pattern"
    elif n_extinct > 0:
        verdict = (
            f"C3a_signal_partial_with_{n_extinct}_extinction"
            f"_of_{n_valid}"
        )
    else:
        verdict = "C3a_insufficient_signal"

    return {
        "n_valid_seeds": n_valid,
        "n_total_seeds": len(seeds_data),
        "aggregate_metrics": agg,
        "patterns_count_across_seeds": {
            "cooperation_apprenable": n_apprenable,
            "cooperation_protocol_emergent": n_protocol,
            "cooperation_cascade_attractor": n_cascade,
            "no_extinction": n_no_ext,
            "clustering_strong_with_50_succ": n_clustering_strong,
        },
        "verdict_global": verdict,
        "seeds": seeds_data,
    }


# ─── Baselines C3a' (smoke 15k, seed=42, max_pop=100) ─────────────────
# Ces valeurs servent de comparateur pour mesurer l'effet de max_pop=60.
# Prédictions user (2026-05-25) si la densité était le bon levier :
#   entropy : 1.37 → ~1.1   (condensation dialecte)
#   dom_sh  : 30.6 % → 45-55 %   (convention locale émergente)
#   delay   : +0.10 → < 0     (anticipation coordinative)
#   cl_trd  : -4.08 → > 0     (convergence apprise)
#   casc    : 17.1 % → > 20 %  (attracteur)
C3A_PRIME_BASELINE = {
    "clustering_mean": 23.22,
    "clustering_trend": -4.08,
    "delay_trend": +0.10,
    "delay_coverage": 0.90,
    "token_dominant_share": 0.306,
    "token_entropy": 1.370,
    "cascade_ratio": 0.171,
    "success_rate": 0.00118,
    "gather_successes": 210,
}


def _delta_vs_baseline(metric_key: str, value: float) -> str:
    """Formate un delta vs baseline C3a' (max_pop=100)."""
    bl = C3A_PRIME_BASELINE.get(metric_key)
    if bl is None or value is None:
        return ""
    delta = value - bl
    direction = "+" if delta >= 0 else ""
    return f" (vs C3a'={bl:.3f}, Δ={direction}{delta:.3f})"


def _print_report(out: dict[str, Any]) -> None:
    print("=" * 70)
    print("AGGREGATE V8-C3a''-soft (multi-seed)")
    print("=" * 70)
    print(f"\nSeeds valides : {out['n_valid_seeds']}/{out['n_total_seeds']}")
    print(f"\n--- Tableau par seed ---")
    print(
        f"{'seed':>6} {'alive':>6} {'births':>7} {'succ':>5} {'rate':>7} "
        f"{'cl_mean':>8} {'cl_trd':>7} {'dom_sh':>7} {'casc_r':>7}"
    )
    for s in out["seeds"]:
        if "error" in s:
            print(f"  {s['run_dir']:<60}  ERROR: {s['error']}")
            continue
        seed = s.get("seed", "?")
        alive = s.get("n_alive_final") or 0
        births = s.get("n_births_total") or 0
        succ = s.get("gather_successes") or 0
        rate = s.get("success_rate") or 0
        cl_m = s.get("clustering_mean") or 0
        cl_t = s.get("clustering_trend") or 0
        dom = s.get("token_dominant_share") or 0
        cas = s.get("cascade_ratio") or 0
        print(
            f"{seed:>6} {alive:>6} {births:>7} {succ:>5} {rate:>6.2%} "
            f"{cl_m:>8.2f} {cl_t:>+7.2f} {dom:>6.1%} {cas:>6.1%}"
        )

    print("\n--- Stats agrégées (mean ± std) ---")
    for k, v in out["aggregate_metrics"].items():
        if v["mean"] is None:
            continue
        print(
            f"  {k:>22} : mean={v['mean']:.3f}  std={v['std']:.3f}  "
            f"min={v['min']:.3f}  max={v['max']:.3f}"
        )

    print("\n--- Patterns détectés (sur N seeds) ---")
    for k, v in out["patterns_count_across_seeds"].items():
        print(f"  [{v}/{out['n_valid_seeds']}] {k}")

    # ─── Prédictions user vs réalité (les 3 signaux à surveiller) ──────
    print("\n--- Prédictions user vs réalité (3 signaux critiques) ---")
    agg = out["aggregate_metrics"]

    ent_mean = agg["token_entropy"]["mean"]
    if ent_mean is not None:
        bl = C3A_PRIME_BASELINE["token_entropy"]
        delta = ent_mean - bl
        verdict = (
            "CONDENSATION DIALECTE" if delta < -0.15 else
            "léger" if delta < -0.05 else
            "pas d'effet"
        )
        print(
            f"  [1] entropy : {bl:.3f} (C3a') → {ent_mean:.3f} "
            f"(Δ={delta:+.3f}) — {verdict}"
        )

    dom_mean = agg["token_dominant_share"]["mean"]
    if dom_mean is not None:
        bl = C3A_PRIME_BASELINE["token_dominant_share"]
        delta = dom_mean - bl
        verdict = (
            "CONVENTION LOCALE" if dom_mean > 0.45 else
            "spécialisation modeste" if delta > 0.05 else
            "pas de spécialisation"
        )
        print(
            f"  [2] dominant_share : {bl:.1%} (C3a') → {dom_mean:.1%} "
            f"(Δ={delta:+.3f}) — {verdict}"
        )

    delay_mean = agg["delay_trend"]["mean"]
    if delay_mean is not None:
        bl = C3A_PRIME_BASELINE["delay_trend"]
        delta = delay_mean - bl
        verdict = (
            "ANTICIPATION COORD." if delay_mean < 0 else
            "stagnation"
        )
        print(
            f"  [3] delay_trend : {bl:+.2f} (C3a') → {delay_mean:+.2f} "
            f"(Δ={delta:+.3f}) — {verdict}"
        )

    cl_mean = agg["clustering_mean"]["mean"]
    cl_trend_mean = agg["clustering_trend"]["mean"]
    if cl_mean is not None:
        bl_m = C3A_PRIME_BASELINE["clustering_mean"]
        bl_t = C3A_PRIME_BASELINE["clustering_trend"]
        print(
            f"  [bonus] clustering_mean : {bl_m:.1f} (C3a') → {cl_mean:.1f} "
            f"  |  trend : {bl_t:+.2f} → {cl_trend_mean:+.2f}"
        )

    cas_mean = agg["cascade_ratio"]["mean"]
    if cas_mean is not None:
        bl = C3A_PRIME_BASELINE["cascade_ratio"]
        delta = cas_mean - bl
        verdict = (
            "ATTRACTEUR" if cas_mean > 0.20 else
            "amélioration" if delta > 0.05 else
            "stable"
        )
        print(
            f"  [bonus] cascade_ratio : {bl:.1%} (C3a') → {cas_mean:.1%} "
            f"(Δ={delta:+.3f}) — {verdict}"
        )

    print(f"\nVerdict global : {out['verdict_global']}")
    print("=" * 70)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="results/v8c3_aggregate.json")
    args = p.parse_args()
    out = aggregate(args.runs)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    _print_report(out)
    print(f"\nWritten : {args.out}")


if __name__ == "__main__":
    main()
