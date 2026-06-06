"""OBS Viewer 2.0 — assemble un report dict Historien depuis un env V8 LIVE.

Réutilise les builders existants (coop_metrics.finalize, spatial_mobility) et
reproduit le petit calcul language_metrics de overnight_v8b1 (sans modifier le
runner taggé). Le DiscoveriesDetector retourne [] pour les blocs absents.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from aetherlife.historian.spatial_mobility import build_spatial_mobility_block


def _language_metrics(env, policy, n_ticks: int) -> dict[str, Any]:
    """Reproduit overnight_v8b1 §language_metrics depuis le registry vocab."""
    if not env.cfg.vocabulary.enabled:
        return {}
    brains = [b for b in policy.registry if b.vocabulary is not None]
    if not brains:
        return {}
    n_tokens = env.cfg.vocabulary.n_tokens
    total = sum(int(b.vocabulary.usage_count.sum()) for b in brains)
    concentrations = []
    for tok in range(n_tokens):
        per = {b.root_id: int(b.vocabulary.usage_count[tok]) for b in brains}
        s = sum(per.values())
        if s > 0:
            concentrations.append(max(per.values()) / s)
    mean_conc = float(np.mean(concentrations)) if concentrations else 0.0
    entropies = [b.vocabulary.usage_entropy() for b in brains]
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    distances = [
        brains[i].vocabulary.distance_to(brains[j].vocabulary)
        for i in range(len(brains)) for j in range(i + 1, len(brains))
    ]
    mean_dist = float(np.mean(distances)) if distances else 0.0
    per_token_top = {
        str(t): int(sum(b.vocabulary.usage_count[t] for b in brains))
        for t in range(n_tokens)
    }
    return {
        "n_brains_with_vocab": len(brains),
        "total_vocalize_count": total,
        "tokens_per_1000_ticks": 1000 * total / max(n_ticks, 1),
        "vocalize_energy_cost_total": (
            total * env.cfg.vocabulary.vocalize_energy_cost
        ),
        "entropy_ratio": mean_entropy / max(float(np.log(n_tokens)), 1e-9),
        "mean_usage_entropy": mean_entropy,
        "mean_token_lineage_concentration": mean_conc,
        "mean_inter_lineage_distance": mean_dist,
        "per_token_usage_top": per_token_top,
    }


def build_live_report(env, policy, occ_start, occ_end, *,
                      windows: tuple, n_ticks: int,
                      seed: int | None = None) -> dict[str, Any]:
    """Report dict consommable par Historian/DiscoveriesDetector (live, MVP)."""
    alive = [a for a in env._agents if a.alive]  # noqa: SLF001
    lin_counts = Counter(a.root_ancestor_id for a in alive)
    n_alive = env.n_alive
    top = [
        {"root_id": r, "alive": c, "pct": 100 * c / max(n_alive, 1)}
        for r, c in lin_counts.most_common(5)
    ]
    aff = Counter(a.biome_affinity for a in alive)
    n_founders = env.cfg.n_agents
    swin, ewin = windows
    return {
        "config": {"seed": seed, "n_ticks": n_ticks, "device": "live",
                   "vision_radius": None},
        "final_state": {
            "n_alive": n_alive,
            "n_births_total": env.n_births_total,
            "n_deaths": env.n_births_total + n_founders - n_alive,
            "top_lineages": top,
            "affinity_distribution": {str(k): v for k, v in aff.items()},
            "n_affinities_alive": len(aff),
        },
        "criterion_3_selection": {
            "n_lineages_initial": n_founders,
            "n_lineages_final": len(policy.registry),
            "dominant_lineage_pct": (top[0]["pct"] if top else 0.0),
        },
        "language_metrics_v8b2": _language_metrics(env, policy, n_ticks),
        "cooperative_v8c3": {
            "enabled": bool(env.cfg.cooperative.enabled),
            "gather_successes_total": int(env.gather_successes_total),
            "gather_failures_total": int(env.gather_failures_total),
        },
        "cooperative_metrics_v8c3": (
            env.coop_metrics.finalize() if env.cfg.cooperative.enabled else {}
        ),
        "spatial_mobility_v8c3": build_spatial_mobility_block(
            occ_start, occ_end, start_window=swin, end_window=ewin,
        ),
    }
