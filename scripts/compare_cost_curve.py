"""V8-C3 phase M — Courbe coût/émergence du langage.

Pour chaque valeur de vocalize_energy_cost ∈ {0.001, 0.01, 0.05, ...},
agrège sur N seeds (good seeds : 42, 99, 256) et calcule :

  cl_trend ctrl       - apprentissage spatial du langage utile
  dom_share ctrl      - discrimination dans l'usage des tokens
  entropy ctrl        - concentration vs bruit
  vocalize_total ctrl - quantité globale d'utilisation
  gather ctrl         - effet net coopération
  Δcl (ablation)      - sensibilité à l'ablation (causalité)

Détection automatique :
  - courbe en cloche : finding majeur signalisation coûteuse
  - courbe monotone : finding linéaire
  - plateau : faible support théorique

Usage :
    python scripts/compare_cost_curve.py \\
        --costs 0.001 0.01 0.05 \\
        --seeds 42 99 256 \\
        --ctrl-dir-template "results/v8c3M_{tag}_ctrl_seed{seed}" \\
        --abl-dir-template "results/v8c3M_{tag}_abl_seed{seed}" \\
        --out results/v8c3_cost_curve.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any


def _load_report(run_dir: str) -> dict[str, Any]:
    for fname in os.listdir(run_dir):
        if fname.startswith("overnight_v8b1_seed") and fname.endswith(".json"):
            with open(os.path.join(run_dir, fname), encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No report in {run_dir}")


def _cost_tag(cost: float) -> str:
    """Convertit un coût en tag de répertoire (ex: 0.001 → 'v001')."""
    # 0.001 → 'M' (run M = baseline), 0.01 → 'v01', 0.03 → 'v03', 0.05 → ''
    if cost == 0.001:
        return "M"  # historique : v8c3M_ctrl_seedXX
    elif cost == 0.01:
        return "M_v01"
    elif cost == 0.03:
        return "M_v03"
    elif cost == 0.05:
        return "soft"  # historique : v8c3a2soft_seedXX
    else:
        return f"M_v{int(cost * 100):02d}"


def _resolve_dir(template: str, tag: str, seed: int) -> str:
    """Substitue {tag} et {seed} dans le template de répertoire."""
    return template.replace("{tag}", tag).replace("{seed}", str(seed))


# ─── Mapping legacy : les runs vcost=0.05 antérieurs sont stockés dans
# des répertoires hérités d'avant le CLI --vocalize-cost. On les map
# explicitement ici pour ne pas casser la rétro-compatibilité.
LEGACY_DIR_OVERRIDE = {
    # (cost, kind) → template explicite
    (0.05, "ctrl"): "results/v8c3a2soft_seed{seed}",
    (0.05, "abl"): "results/v8c3a2soft_ablation_seed{seed}",
}


def _resolve_dir_with_legacy(
    template: str, tag: str, seed: int, cost: float, kind: str,
) -> str:
    """Resolve avec fallback sur LEGACY_DIR_OVERRIDE pour les coûts hérités."""
    override = LEGACY_DIR_OVERRIDE.get((cost, kind))
    if override:
        return override.replace("{seed}", str(seed))
    return _resolve_dir(template, tag, seed)


def _extract(report: dict[str, Any]) -> dict[str, Any]:
    coop = report.get("cooperative_v8c3", {}) or {}
    m = report.get("cooperative_metrics_v8c3", {}) or {}
    fs = report.get("final_state", {}) or {}
    lang = report.get("language_metrics_v8b2", {}) or {}
    cl = m.get("clustering_pre_success", {}) or {}
    tk = m.get("token_entropy_pre_success", {}) or {}
    return {
        "n_alive": fs.get("n_alive"),
        "n_births": fs.get("n_births_total"),
        "gather_successes": coop.get("gather_successes_total", 0),
        "vocalize_total": lang.get("total_vocalize_count", 0),
        "cl_trend": cl.get("trend_q4_minus_q1"),
        "cl_mean": cl.get("mean_neighbors_r3"),
        "dom_share": tk.get("dominant_share"),
        "dominant_token": tk.get("dominant_token"),
        "entropy": tk.get("entropy"),
    }


def _agg(values: list) -> dict[str, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    if len(vals) == 1:
        return {"mean": float(vals[0]), "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.stdev(vals)),
        "n": len(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _detect_shape(costs: list[float], values: list[float | None]) -> str:
    """Détecte la forme de la courbe : monotone / cloche / plateau."""
    vals = [
        v for v in values if v is not None
    ]
    if len(vals) < 3:
        return "insufficient_data"
    sorted_pairs = sorted(zip(costs, values))
    valid = [(c, v) for c, v in sorted_pairs if v is not None]
    if len(valid) < 3:
        return "insufficient_data"
    deltas = [
        valid[i + 1][1] - valid[i][1] for i in range(len(valid) - 1)
    ]
    n_pos = sum(1 for d in deltas if d > 0.3)
    n_neg = sum(1 for d in deltas if d < -0.3)
    n_flat = sum(1 for d in deltas if abs(d) <= 0.3)
    if n_pos == len(deltas) or n_neg == len(deltas):
        return "monotone"
    if n_pos > 0 and n_neg > 0:
        return "bell_curve"
    if n_flat == len(deltas):
        return "plateau"
    return "irregular"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--costs", nargs="+", type=float, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument(
        "--ctrl-dir-template", required=True,
        help="ex: 'results/v8c3{tag_prefix}_ctrl_seed{seed}'",
    )
    p.add_argument(
        "--abl-dir-template", required=True,
        help="ex: 'results/v8c3{tag_prefix}_abl_seed{seed}'",
    )
    p.add_argument("--out", default="results/v8c3_cost_curve.json")
    args = p.parse_args()

    curve = []
    for cost in args.costs:
        tag = _cost_tag(cost)
        ctrl_values = []
        abl_values = []
        for seed in args.seeds:
            ctrl_dir = _resolve_dir_with_legacy(
                args.ctrl_dir_template, tag, seed, cost, "ctrl",
            )
            abl_dir = _resolve_dir_with_legacy(
                args.abl_dir_template, tag, seed, cost, "abl",
            )
            try:
                ctrl_r = _load_report(ctrl_dir)
                ctrl_d = _extract(ctrl_r)
                ctrl_d["seed"] = seed
                ctrl_values.append(ctrl_d)
            except FileNotFoundError:
                pass
            try:
                abl_r = _load_report(abl_dir)
                abl_d = _extract(abl_r)
                abl_d["seed"] = seed
                abl_values.append(abl_d)
            except FileNotFoundError:
                pass

        point = {
            "vocalize_cost": cost,
            "tag": tag,
            "n_ctrl_seeds": len(ctrl_values),
            "n_abl_seeds": len(abl_values),
            "ctrl_per_seed": ctrl_values,
            "abl_per_seed": abl_values,
            "ctrl_agg": {
                k: _agg([s[k] for s in ctrl_values])
                for k in [
                    "cl_trend", "cl_mean", "dom_share", "entropy",
                    "gather_successes", "vocalize_total",
                    "n_births", "n_alive",
                ]
            },
        }
        # Δcl(ablation) = abl.cl_trend - ctrl.cl_trend par seed
        deltas = []
        for ctrl_s in ctrl_values:
            seed = ctrl_s["seed"]
            for abl_s in abl_values:
                if abl_s["seed"] == seed:
                    if ctrl_s["cl_trend"] is not None and abl_s["cl_trend"] is not None:
                        deltas.append(abl_s["cl_trend"] - ctrl_s["cl_trend"])
                    break
        point["delta_cl_ablation_agg"] = _agg(deltas)
        point["delta_cl_per_seed"] = deltas
        curve.append(point)

    # Détection shape sur cl_trend ctrl
    costs_sorted = [p["vocalize_cost"] for p in curve]
    cl_means = [p["ctrl_agg"]["cl_trend"]["mean"] for p in curve]
    shape_cl = _detect_shape(costs_sorted, cl_means)

    # Détection shape sur Δcl ablation (effet causal)
    delta_means = [p["delta_cl_ablation_agg"]["mean"] for p in curve]
    shape_delta = _detect_shape(costs_sorted, delta_means)

    # Détection shape sur dom_share
    dom_means = [p["ctrl_agg"]["dom_share"]["mean"] for p in curve]
    shape_dom = _detect_shape(costs_sorted, dom_means)

    # Verdict global
    if shape_cl == "bell_curve":
        verdict = "FINDING_MAJEUR_SIGNALISATION_COUTEUSE"
    elif shape_cl == "monotone":
        verdict = "FINDING_LINEAIRE_COUT_DISCRIMINANT"
    elif shape_cl == "plateau":
        verdict = "EFFET_SATURE_PAS_DE_DYNAMIQUE_DE_COUT"
    else:
        verdict = f"shape={shape_cl}_a_inspecter"

    out = {
        "costs_tested": costs_sorted,
        "seeds_used": args.seeds,
        "curve_points": curve,
        "shapes": {
            "cl_trend_ctrl": shape_cl,
            "delta_cl_ablation": shape_delta,
            "dom_share_ctrl": shape_dom,
        },
        "verdict": verdict,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    # Console
    print("=" * 78)
    print(f"V8-C3 phase M — COURBE COÛT/ÉMERGENCE ({len(args.seeds)} seeds)")
    print("=" * 78)
    print(f"\nSeeds : {args.seeds}")
    print(f"Coûts testés : {costs_sorted}")
    print()
    print(
        f"{'vcost':>7} {'n_ctrl':>6} {'cl_t_m':>8} {'cl_t_std':>9} "
        f"{'dom_sh':>7} {'entropy':>8} "
        f"{'g_succ':>7} {'voc_tot':>9} {'Δcl_abl':>9}"
    )
    for p in curve:
        cl = p["ctrl_agg"]["cl_trend"]
        dom = p["ctrl_agg"]["dom_share"]
        ent = p["ctrl_agg"]["entropy"]
        gs = p["ctrl_agg"]["gather_successes"]
        vt = p["ctrl_agg"]["vocalize_total"]
        d = p["delta_cl_ablation_agg"]
        print(
            f"{p['vocalize_cost']:>7.3f} {p['n_ctrl_seeds']:>6} "
            f"{cl['mean'] if cl['mean'] is not None else 0:>+8.2f} "
            f"{cl['std'] if cl['std'] is not None else 0:>9.2f} "
            f"{dom['mean'] if dom['mean'] is not None else 0:>7.2%} "
            f"{ent['mean'] if ent['mean'] is not None else 0:>8.3f} "
            f"{gs['mean'] if gs['mean'] is not None else 0:>7.0f} "
            f"{vt['mean'] if vt['mean'] is not None else 0:>9.0f} "
            f"{d['mean'] if d['mean'] is not None else 0:>+9.2f}"
        )

    print(f"\n--- Formes des courbes ---")
    print(f"  cl_trend ctrl     : {shape_cl}")
    print(f"  Δcl ablation      : {shape_delta}")
    print(f"  dom_share ctrl    : {shape_dom}")
    print(f"\nVerdict : {verdict}")
    print("=" * 78)
    print(f"\nWritten : {args.out}")


if __name__ == "__main__":
    main()
