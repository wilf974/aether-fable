"""V8-C3 — Comparaison multi-seed témoin vs ablation langage sur "good seeds".

Suit le pivot méthodologique : on n'évalue l'ablation que sur les seeds
qui ont franchi `cooperation_apprenable` (sentinel critère d'entrée).
Sur les "bad seeds" (mécanique non maîtrisée), l'ablation mesurerait du
bruit, pas une fonction du langage.

Hypothèse causale H1 : si le langage a une fonction coopérative émergente
chez les good seeds, alors désactiver vocalize @ t=10k doit faire chuter
gather_successes APRES t=10k DIFFERENTIELLEMENT vs témoin.

Hypothèse nulle H0 : le langage est décoratif même chez les good seeds.

Sortie : Δ % naissances, gather_successes_post, clustering trend, etc.
        + verdict probabiliste (causal_strong / partial / null).

Usage :
    python scripts/compare_good_seeds_ablation.py \\
        --control-dir-prefix results/v8c3a2soft_seed \\
        --ablation-dir-prefix results/v8c3a2soft_ablation_seed \\
        --seeds 42 99 256 2048 \\
        --ablation-tick 10000 \\
        --out results/v8c3_ablation_compare.json
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


def _coop(r: dict[str, Any]) -> dict[str, Any]:
    return r.get("cooperative_v8c3", {}) or {}


def _metrics(r: dict[str, Any]) -> dict[str, Any]:
    return r.get("cooperative_metrics_v8c3", {}) or {}


def _split_curve_at(
    curve: list, t_split: int,
) -> tuple[list, list]:
    """Retourne (before, after) au-delà de t_split inclus."""
    before, after = [], []
    for t, v in curve or []:
        if t < t_split:
            before.append((t, v))
        else:
            after.append((t, v))
    return before, after


def _mean_after(curve_after: list) -> float:
    if not curve_after:
        return 0.0
    return sum(v for _, v in curve_after) / len(curve_after)


def _last_alive(curve: list) -> int:
    if not curve:
        return 0
    return int(curve[-1][1])


def _pct_delta(ctrl: float, abl: float) -> float:
    if ctrl == 0:
        return 0.0
    return 100.0 * (abl - ctrl) / ctrl


def _compare_seed(
    ctrl: dict[str, Any], abl: dict[str, Any], t_split: int,
) -> dict[str, Any]:
    fs_c = ctrl.get("final_state", {}) or {}
    fs_a = abl.get("final_state", {}) or {}
    coop_c = _coop(ctrl)
    coop_a = _coop(abl)
    m_c = _metrics(ctrl)
    m_a = _metrics(abl)

    # Population
    alive_c = fs_c.get("n_alive", 0) or 0
    alive_a = fs_a.get("n_alive", 0) or 0
    births_c = fs_c.get("n_births_total", 0) or 0
    births_a = fs_a.get("n_births_total", 0) or 0

    # Cooperation
    succ_c = coop_c.get("gather_successes_total", 0) or 0
    succ_a = coop_a.get("gather_successes_total", 0) or 0

    # Curves "après ablation"
    alive_curve_c = (ctrl.get("curves", {}) or {}).get("alive") or []
    alive_curve_a = (abl.get("curves", {}) or {}).get("alive") or []
    _, alive_after_c = _split_curve_at(alive_curve_c, t_split)
    _, alive_after_a = _split_curve_at(alive_curve_a, t_split)

    # Cooperative metrics (déjà agrégées sur le run)
    cl_c = (m_c.get("clustering_pre_success") or {}).get(
        "trend_q4_minus_q1"
    ) or 0
    cl_a = (m_a.get("clustering_pre_success") or {}).get(
        "trend_q4_minus_q1"
    ) or 0
    delay_c = (m_c.get("vocalize_to_gather_delay") or {}).get(
        "trend_q4_minus_q1"
    ) or 0
    delay_a = (m_a.get("vocalize_to_gather_delay") or {}).get(
        "trend_q4_minus_q1"
    ) or 0
    dom_c = (m_c.get("token_entropy_pre_success") or {}).get(
        "dominant_share"
    ) or 0
    dom_a = (m_a.get("token_entropy_pre_success") or {}).get(
        "dominant_share"
    ) or 0

    return {
        # Pop
        "alive_ctrl": alive_c, "alive_abl": alive_a,
        "alive_delta_pct": _pct_delta(alive_c, alive_a),
        "births_ctrl": births_c, "births_abl": births_a,
        "births_delta_pct": _pct_delta(births_c, births_a),
        "alive_after_mean_ctrl": _mean_after(alive_after_c),
        "alive_after_mean_abl": _mean_after(alive_after_a),
        # Cooperation (le critère le plus important)
        "gather_ctrl": succ_c, "gather_abl": succ_a,
        "gather_delta_pct": _pct_delta(succ_c, succ_a),
        # Métriques coop
        "cl_trend_ctrl": cl_c, "cl_trend_abl": cl_a,
        "cl_trend_delta": cl_a - cl_c,
        "delay_trend_ctrl": delay_c, "delay_trend_abl": delay_a,
        "delay_trend_delta": delay_a - delay_c,
        "dom_share_ctrl": dom_c, "dom_share_abl": dom_a,
        "dom_share_delta": dom_a - dom_c,
    }


def _verdict(seeds_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Verdict probabiliste sur N seeds.

    Le critère causal CENTRAL : gather_successes_total chute-t-il dans
    l'ablation vs témoin ? Sur les good seeds, on s'attend à chute > 20 %
    si H1, < 5 % si H0, intermédiaire si décoratif partiel.
    """
    gather_deltas = [
        s["gather_delta_pct"] for s in seeds_data
        if s.get("gather_ctrl", 0) >= 30
    ]
    births_deltas = [s["births_delta_pct"] for s in seeds_data]
    alive_deltas = [s["alive_delta_pct"] for s in seeds_data]

    def _agg(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
        if len(vals) == 1:
            return {
                "mean": vals[0], "std": 0, "min": vals[0],
                "max": vals[0], "n": 1,
            }
        return {
            "mean": float(statistics.mean(vals)),
            "std": float(statistics.stdev(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n": len(vals),
        }

    g = _agg(gather_deltas)
    b = _agg(births_deltas)
    a = _agg(alive_deltas)

    # Verdict basé sur gather_delta (le test causal central)
    if g["mean"] <= -20 and g["n"] >= 2:
        v = "communication_causale_renforcee"
    elif g["mean"] <= -10 and g["n"] >= 2:
        v = "communication_partielle_significative"
    elif -10 < g["mean"] < +5:
        v = "communication_decorative_hypothese_tient"
    elif g["mean"] >= +5:
        v = "ablation_renforce_paradoxal"
    else:
        v = "ambigu_plus_de_donnees"

    return {
        "gather_delta_pct_agg": g,
        "births_delta_pct_agg": b,
        "alive_delta_pct_agg": a,
        "verdict": v,
        "n_good_seeds_compared": len(seeds_data),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--control-dir-prefix", required=True,
        help="ex: results/v8c3a2soft_seed",
    )
    p.add_argument(
        "--ablation-dir-prefix", required=True,
        help="ex: results/v8c3a2soft_ablation_seed",
    )
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--ablation-tick", type=int, required=True)
    p.add_argument(
        "--out", default="results/v8c3_ablation_compare.json",
    )
    args = p.parse_args()

    seeds_data = []
    for seed in args.seeds:
        ctrl_dir = f"{args.control_dir_prefix}{seed}"
        abl_dir = f"{args.ablation_dir_prefix}{seed}"
        try:
            ctrl = _load_report(ctrl_dir)
            abl = _load_report(abl_dir)
        except FileNotFoundError as e:
            print(f"[seed {seed}] SKIP: {e}")
            continue
        d = _compare_seed(ctrl, abl, args.ablation_tick)
        d["seed"] = seed
        seeds_data.append(d)

    verdict_block = _verdict(seeds_data)

    out = {
        "ablation_tick": args.ablation_tick,
        "seeds_compared": [s["seed"] for s in seeds_data],
        "seeds_data": seeds_data,
        **verdict_block,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Console report
    print("=" * 78)
    print(
        f"V8-C3 ABLATION COMPARE (good seeds, vocalize désactivé "
        f"@ t={args.ablation_tick})"
    )
    print("=" * 78)
    print(
        f"\nSeeds comparés : {len(seeds_data)} "
        f"({', '.join(str(s['seed']) for s in seeds_data)})"
    )

    print("\n--- Par seed (témoin → ablation) ---")
    print(
        f"{'seed':>5} {'g_ctrl':>7} {'g_abl':>6} {'Δg%':>7} "
        f"{'b_ctrl':>7} {'b_abl':>7} {'Δb%':>7} "
        f"{'a_ctrl':>6} {'a_abl':>6} {'Δa%':>7}"
    )
    for s in seeds_data:
        print(
            f"{s['seed']:>5} {s['gather_ctrl']:>7} {s['gather_abl']:>6} "
            f"{s['gather_delta_pct']:>+6.1f}% "
            f"{s['births_ctrl']:>7} {s['births_abl']:>7} "
            f"{s['births_delta_pct']:>+6.1f}% "
            f"{s['alive_ctrl']:>6} {s['alive_abl']:>6} "
            f"{s['alive_delta_pct']:>+6.1f}%"
        )

    print("\n--- Métriques coop (témoin → ablation, deltas absolus) ---")
    print(
        f"{'seed':>5} {'cl_t_c':>7} {'cl_t_a':>7} {'Δcl':>6} "
        f"{'dly_c':>6} {'dly_a':>6} {'Δdly':>6} "
        f"{'dm_c':>5} {'dm_a':>5} {'Δdm':>6}"
    )
    for s in seeds_data:
        print(
            f"{s['seed']:>5} "
            f"{s['cl_trend_ctrl']:>+7.2f} {s['cl_trend_abl']:>+7.2f} "
            f"{s['cl_trend_delta']:>+6.2f} "
            f"{s['delay_trend_ctrl']:>+6.2f} {s['delay_trend_abl']:>+6.2f} "
            f"{s['delay_trend_delta']:>+6.2f} "
            f"{s['dom_share_ctrl']:>5.2f} {s['dom_share_abl']:>5.2f} "
            f"{s['dom_share_delta']:>+6.2f}"
        )

    print("\n--- Agrégats des deltas (%, mean ± std) ---")
    g = verdict_block["gather_delta_pct_agg"]
    b = verdict_block["births_delta_pct_agg"]
    a = verdict_block["alive_delta_pct_agg"]
    print(
        f"  gather_delta_pct : mean={g['mean']:+.1f}% "
        f"std={g['std']:.1f} min={g['min']:+.1f} max={g['max']:+.1f} "
        f"(n_good={g['n']})"
    )
    print(
        f"  births_delta_pct : mean={b['mean']:+.1f}% "
        f"std={b['std']:.1f} min={b['min']:+.1f} max={b['max']:+.1f}"
    )
    print(
        f"  alive_delta_pct  : mean={a['mean']:+.1f}% "
        f"std={a['std']:.1f} min={a['min']:+.1f} max={a['max']:+.1f}"
    )

    print(f"\nVerdict : {verdict_block['verdict']}")
    print("=" * 78)
    print(f"\nWritten : {args.out}")


if __name__ == "__main__":
    main()
