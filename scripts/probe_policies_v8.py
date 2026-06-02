"""OBS V3.0 — capture : run un seed, calcule mobility_score, sonde le cerveau
de la lignée dominante survivante → JSON. Réutilise build_env (runner intact).

Usage:
    python scripts/probe_policies_v8.py --seed 25 --ticks 16000 \
        --device cuda --out results/probe/seed25.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overnight_v8b1 import build_env  # noqa: E402
from aetherlife.agents.lineage_agent import LineageAgent  # noqa: E402
from aetherlife.agents.lineage_brain import BrainConfig  # noqa: E402
from aetherlife.historian.spatial_mobility import (  # noqa: E402
    OccupancyAccumulator, build_spatial_mobility_block, window_bounds,
)
from aetherlife.viz.policy_probe import (  # noqa: E402
    ACTION_LABELS, PROBE_LABELS, fingerprint, make_probe_env,
)


def probe(seed: int, *, ticks: int = 16000,
          regime: str = "coordination_collective",
          device: str = "cuda", out: str = "results/probe/seed.json") -> str:
    env = build_env(seed, regime=regime, vocalize_energy_cost=0.05)
    vision_radius = 2 if regime in (
        "coordination", "coordination_hidden", "coordination_hard",
    ) else 4
    cfg = BrainConfig(
        enabled=True, device=device, vision_radius=vision_radius,
        hidden_dims=(64, 64), lr=1e-4, batch_size=64,
        buffer_capacity=50_000, min_replay_to_learn=500, train_every=4,
        epsilon_start=0.6, epsilon_end=0.08, epsilon_decay_steps=30_000,
        target_sync_steps=200, mutation_std=0.03,
    )
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=seed)

    swin, ewin = window_bounds(ticks)
    occ_s = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    occ_e = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    for t in range(1, ticks + 1):
        if env.n_alive == 0:
            break
        obs_stub = {a.agent_id: np.zeros(10)
                    for a in env._agents if a.alive}  # noqa: SLF001
        actions = policy.act_dict(obs_stub, greedy=False)
        env.step(actions)
        if swin[0] < t <= swin[1] or ewin[0] < t <= ewin[1]:
            pos = [(a.pos[0], a.pos[1])
                   for a in env._agents if a.alive]  # noqa: SLF001
            (occ_s if t <= swin[1] else occ_e).add_positions(pos)

    sm = build_spatial_mobility_block(occ_s, occ_e,
                                      start_window=swin, end_window=ewin)

    # lignée dominante survivante
    alive_roots = Counter(
        a.root_ancestor_id for a in env._agents if a.alive  # noqa: SLF001
    )
    if not alive_roots:
        raise RuntimeError(f"seed {seed} : extinction, pas de cerveau à sonder")
    dom_root = alive_roots.most_common(1)[0][0]
    brain = policy.registry.get(dom_root)
    if brain is None:
        raise RuntimeError(f"seed {seed} : brain dominant {dom_root} introuvable")

    penv = make_probe_env(seed=seed)
    fp = fingerprint(brain, penv)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    record = {
        "seed": seed, "ticks": ticks, "regime": regime,
        "mobility_score": sm["corr_occupation_start_end"],
        "village_basin": sm["village_basin"],
        "dominant_lineage": int(dom_root),
        "n_alive_final": int(env.n_alive),
        "action_labels": ACTION_LABELS,
        "probe_labels": PROBE_LABELS,
        "fingerprint": fp.tolist(),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f)
    print(f"WROTE {out}  mobility={record['mobility_score']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ticks", type=int, default=16000)
    p.add_argument("--regime", default="coordination_collective")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/probe/seed.json")
    a = p.parse_args()
    probe(a.seed, ticks=a.ticks, regime=a.regime, device=a.device, out=a.out)


if __name__ == "__main__":
    main()
