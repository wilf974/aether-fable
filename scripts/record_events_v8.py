"""Recorder — re-simule un seed V8-C3 et dumpe events.jsonl + meta.json.

Réutilise build_env de overnight_v8b1 SANS le modifier. Reproduit verbatim le
setup BrainConfig + vision_radius de overnight_v8b1.run.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/record_events_v8.py \\
        --seed 25 --ticks 16000 --record-every 10 \\
        --regime coordination_collective --device cuda --out-dir results/clip_seed25
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overnight_v8b1 import build_env  # noqa: E402
from aetherlife.agents.lineage_agent import LineageAgent  # noqa: E402
from aetherlife.agents.lineage_brain import BrainConfig  # noqa: E402


def _spot_adjacency(env, pos) -> int:
    """Nb d'agents vivants à distance Manhattan <= 1 du spot."""
    n = 0
    for a in env._agents:  # noqa: SLF001
        if a.alive and abs(a.pos[0] - pos[0]) + abs(a.pos[1] - pos[1]) <= 1:
            n += 1
    return n


def record(
    seed: int,
    *,
    regime: str = "coordination_collective",
    ticks: int = 16000,
    vocalize_cost: float = 0.05,
    max_pop_override: int | None = None,
    bonus_energy_override: float | None = None,
    record_every: int = 10,
    out_dir: str = "results/clip",
    device: str = "cuda",
) -> str:
    env = build_env(
        seed, regime=regime, vocalize_energy_cost=vocalize_cost,
        max_pop_override=max_pop_override,
        bonus_energy_override=bonus_energy_override,
    )
    # Mirror verbatim de overnight_v8b1.run (lignes 392-402)
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

    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "rows": env.cfg.rows, "cols": env.cfg.cols,
        "n_tokens": env.cfg.vocabulary.n_tokens,
        "listen_radius": env.cfg.vocabulary.listen_radius,
        "seed": seed, "regime": regime, "vcost": vocalize_cost,
        "total_ticks": ticks, "record_every": record_every,
        "schema_version": 1,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    path = os.path.join(out_dir, "events.jsonl")
    with open(path, "w", encoding="utf-8") as out:
        for t in range(1, ticks + 1):
            if env.n_alive == 0:
                break
            obs_stub = {
                a.agent_id: np.zeros(10)
                for a in env._agents if a.alive  # noqa: SLF001
            }
            actions = policy.act_dict(obs_stub, greedy=False)
            env.step(actions)
            if t % record_every != 0:
                continue
            agents = []
            for a in env._agents:  # noqa: SLF001
                if not a.alive:
                    continue
                agents.append({
                    "id": a.agent_id, "lin": a.root_ancestor_id,
                    "r": a.pos[0], "c": a.pos[1],
                    "e": round(float(a.energy), 1),
                    "er": round(float(a.energy) / float(env.cfg.max_energy), 3),
                    "age": t - a.birth_tick,
                    "aff": a.biome_affinity,
                })
            vocal = {
                str(sid): int(tid)
                for sid, tid in getattr(env, "_tokens_this_tick", {}).items()
            }
            spots = [
                {"r": pos[0], "c": pos[1], "n": _spot_adjacency(env, pos)}
                for pos in getattr(env, "gather_spots", [])
            ]
            ev = {
                "t": t, "season": int(env.season), "n_alive": env.n_alive,
                "n_lin": len(policy.registry),
                "agents": agents, "vocal": vocal, "spots": spots,
            }
            out.write(json.dumps(ev) + "\n")
    print(f"WROTE {path}  ({out_dir}/meta.json)")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--regime", default="coordination_collective")
    p.add_argument("--ticks", type=int, default=16000)
    p.add_argument("--vocalize-cost", type=float, default=0.05)
    p.add_argument("--max-pop-override", type=int, default=None)
    p.add_argument("--bonus-energy-override", type=float, default=None)
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--out-dir", default="results/clip")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    record(
        a.seed, regime=a.regime, ticks=a.ticks, vocalize_cost=a.vocalize_cost,
        max_pop_override=a.max_pop_override,
        bonus_energy_override=a.bonus_energy_override,
        record_every=a.record_every, out_dir=a.out_dir, device=a.device,
    )


if __name__ == "__main__":
    main()
