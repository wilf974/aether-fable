"""V8-B2.3 — Comparaison témoin vs ablation interventionnelle du langage.

Lit les 2 JSON résultats (témoin et ablation à t=T) et produit un rapport
comparatif probabiliste : le canal de communication a-t-il une fonction ?

Usage :
    python scripts/compare_ablation.py \\
        --control results/v8b2_3_control/overnight_v8b1_seed42.json \\
        --ablation results/v8b2_3_ablation_15k/overnight_v8b1_seed42.json \\
        --ablation-tick 15000
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _curve_split(curve: list, t_split: int) -> tuple[list, list]:
    """Split une curve [(tick, val), ...] en (avant, après) t_split."""
    before, after = [], []
    for t, v in curve:
        if t < t_split:
            before.append((t, v))
        else:
            after.append((t, v))
    return before, after


def _curve_summary(curve: list) -> dict:
    if not curve:
        return {"n": 0, "mean": 0.0, "last": 0.0}
    vals = [v for _, v in curve]
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "last": float(vals[-1]),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _delta_pct(after_ctrl: float, after_abl: float) -> float:
    """% différence de l'ablation par rapport au témoin."""
    if abs(after_ctrl) < 1e-9:
        return 0.0
    return (after_abl - after_ctrl) / abs(after_ctrl) * 100.0


def compare(control: dict, ablation: dict, t_split: int) -> dict:
    """Compare 2 JSONs résultats avant/après le tick d'ablation."""
    out: dict[str, Any] = {"ablation_tick": t_split}

    # ─── Curves split ──────────────────────────────────────────────────
    for key in ("alive", "lineages", "loss"):
        c_curve = control.get("curves", {}).get(key, [])
        a_curve = ablation.get("curves", {}).get(key, [])
        c_before, c_after = _curve_split(c_curve, t_split)
        a_before, a_after = _curve_split(a_curve, t_split)
        out[f"{key}_before_ctrl"] = _curve_summary(c_before)
        out[f"{key}_before_abl"] = _curve_summary(a_before)
        out[f"{key}_after_ctrl"] = _curve_summary(c_after)
        out[f"{key}_after_abl"] = _curve_summary(a_after)

    # ─── Final state comparison ────────────────────────────────────────
    out["final_alive_ctrl"] = control.get("final_state", {}).get("n_alive", 0)
    out["final_alive_abl"] = ablation.get("final_state", {}).get("n_alive", 0)
    out["final_lineages_ctrl"] = (
        control.get("criterion_3_selection", {}).get("n_lineages_final", 0)
    )
    out["final_lineages_abl"] = (
        ablation.get("criterion_3_selection", {}).get("n_lineages_final", 0)
    )
    out["n_births_ctrl"] = control.get("final_state", {}).get("n_births_total", 0)
    out["n_births_abl"] = ablation.get("final_state", {}).get("n_births_total", 0)

    # ─── Language causality (control only, ablation forcé à 0) ────────
    ctrl_cause = control.get("language_causality_v8b2_2", {})
    abl_cause = ablation.get("language_causality_v8b2_2", {})
    out["shift_mean_ctrl"] = ctrl_cause.get("listener_shift_mean", 0)
    out["shift_mean_abl"] = abl_cause.get("listener_shift_mean", 0)

    # ─── Verdict probabiliste ──────────────────────────────────────────
    out["verdict"] = _verdict(out)
    return out


def _verdict(comp: dict) -> dict:
    """Compute verdict probabiliste."""
    # Δ pop final
    delta_alive = _delta_pct(comp["final_alive_ctrl"], comp["final_alive_abl"])
    # Δ pop moyenne après ablation
    delta_alive_after = _delta_pct(
        comp["alive_after_ctrl"]["mean"], comp["alive_after_abl"]["mean"]
    )
    # Δ births
    delta_births = _delta_pct(comp["n_births_ctrl"], comp["n_births_abl"])
    # Δ lignées finales
    delta_lineages = _delta_pct(comp["final_lineages_ctrl"], comp["final_lineages_abl"])

    metrics = {
        "delta_alive_final_pct": delta_alive,
        "delta_alive_after_mean_pct": delta_alive_after,
        "delta_births_pct": delta_births,
        "delta_lineages_pct": delta_lineages,
    }

    # Décision
    big_drops = sum(
        1 for v in metrics.values() if v <= -20.0
    )
    small_diffs = sum(
        1 for v in metrics.values() if abs(v) <= 5.0
    )

    if big_drops >= 2:
        decision = "communication_utile"
        reason = (
            f"{big_drops}/4 métriques chutent de >20 % après ablation — "
            f"compatible avec hypothèse 'le canal a une fonction'."
        )
    elif small_diffs >= 3:
        decision = "communication_decorative"
        reason = (
            f"{small_diffs}/4 métriques restent dans ±5 % — compatible avec "
            f"hypothèse 'le canal est non-fonctionnel'."
        )
    else:
        decision = "ambigu"
        reason = (
            "Signal mixte. Certaines métriques baissent, d'autres pas. "
            "Plus de seeds ou de runs plus longs recommandés."
        )

    return {"decision": decision, "reason": reason, "deltas": metrics}


def print_report(comp: dict) -> None:
    print("=" * 70)
    print(f"V8-B2.3 — RAPPORT COMPARATIF TÉMOIN vs ABLATION @ t={comp['ablation_tick']}")
    print("=" * 70)

    print(f"\n[POPULATION VIVANTE]")
    print(f"  Avant t={comp['ablation_tick']} :")
    print(f"    Témoin   : mean {comp['alive_before_ctrl']['mean']:.0f}")
    print(f"    Ablation : mean {comp['alive_before_abl']['mean']:.0f}")
    print(f"  Après t={comp['ablation_tick']} :")
    print(f"    Témoin   : mean {comp['alive_after_ctrl']['mean']:.0f}  last {comp['alive_after_ctrl']['last']:.0f}")
    print(f"    Ablation : mean {comp['alive_after_abl']['mean']:.0f}  last {comp['alive_after_abl']['last']:.0f}")
    print(f"  Final :")
    print(f"    Témoin   : {comp['final_alive_ctrl']} vivants")
    print(f"    Ablation : {comp['final_alive_abl']} vivants")

    print(f"\n[LIGNÉES FINALES]")
    print(f"  Témoin   : {comp['final_lineages_ctrl']}")
    print(f"  Ablation : {comp['final_lineages_abl']}")

    print(f"\n[NAISSANCES TOTALES]")
    print(f"  Témoin   : {comp['n_births_ctrl']}")
    print(f"  Ablation : {comp['n_births_abl']}")

    print(f"\n[CAUSALITY KL SHIFT (sanity check)]")
    print(f"  Témoin   : {comp['shift_mean_ctrl']:.4f}")
    print(f"  Ablation : {comp['shift_mean_abl']:.4f}  (attendu < témoin : canal coupé après t={comp['ablation_tick']})")

    print(f"\n[DELTAS (ablation vs témoin)]")
    for k, v in comp["verdict"]["deltas"].items():
        sign = "+" if v >= 0 else ""
        print(f"  {k:35s} : {sign}{v:6.1f} %")

    print(f"\n[VERDICT PROBABILISTE]")
    print(f"  Decision : {comp['verdict']['decision']}")
    print(f"  Reason   : {comp['verdict']['reason']}")

    print(f"\n[LECTURE MÉTHODOLOGIQUE]")
    print(
        "  Ce test ne prouve PAS la sémantique des tokens. Il mesure uniquement\n"
        "  si la disponibilité du canal a un impact observable sur la survie\n"
        "  collective. Une absence d'impact n'exclut pas que le langage existait\n"
        "  (les agents peuvent avoir une stratégie compensatoire). Une présence\n"
        "  d'impact ne prouve pas la compréhension non plus."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--control", required=True, help="Path JSON résultat témoin")
    p.add_argument("--ablation", required=True, help="Path JSON résultat ablation")
    p.add_argument(
        "--ablation-tick", type=int, default=15000,
        help="Tick où l'ablation a été déclenchée",
    )
    p.add_argument(
        "--out-json", default=None,
        help="Si fourni, écrit le dict de comparaison en JSON",
    )
    args = p.parse_args()

    ctrl = _load(args.control)
    abl = _load(args.ablation)
    comp = compare(ctrl, abl, args.ablation_tick)
    print_report(comp)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2, default=str)
        print(f"\nJSON saved : {args.out_json}")


if __name__ == "__main__":
    main()
