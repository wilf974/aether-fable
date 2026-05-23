"""V2 density sweep — démontrer la tragedy of the commons.

Mesure mean_lifespan, alive_rate, food_eaten en fonction de N agents,
avec une politique random (sans entraînement) pour isoler l'effet densité.

Usage :
    python scripts/v2_density_sweep.py
"""
from __future__ import annotations

import argparse
import statistics

import numpy as np

from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)


def run_one_episode(env: MultiAgentFoodGrid, rng: np.random.Generator) -> dict:
    obs_dict, _ = env.reset(seed=int(rng.integers(0, 10**6)))
    lifespans: dict[int, int] = {aid: 0 for aid in env.alive_agent_ids}
    total_food = 0
    while env.n_alive > 0 and env.step_count < env.cfg.max_steps:
        actions = {aid: int(rng.integers(0, 4)) for aid in env.alive_agent_ids}
        _, _, terminated, truncated, infos = env.step(actions)
        for aid in actions:
            if env.agent_state(aid).alive or terminated.get(aid, False) or truncated.get(aid, False):
                lifespans[aid] = env.step_count
            if infos.get(aid, {}).get("ate"):
                total_food += 1
        if all(terminated.get(aid, False) for aid in actions):
            break
    return {
        "n_alive": env.n_alive,
        "alive_rate": env.n_alive / env.cfg.n_agents,
        "mean_lifespan": statistics.mean(lifespans.values()) if lifespans else 0.0,
        "total_food": total_food,
        "n_dead": env.n_dead,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents-sweep", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--episodes-per-n", type=int, default=10)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--initial-food-density", type=float, default=0.05)
    parser.add_argument("--food-respawn-lambda", type=float, default=1.0)
    args = parser.parse_args()

    print(
        f"AetherLife V2 density sweep (random policy) — "
        f"grid={args.rows}x{args.cols} density={args.initial_food_density} "
        f"respawn={args.food_respawn_lambda} max_steps={args.max_steps}"
    )
    print(f"  N sweep: {args.n_agents_sweep}  episodes per N: {args.episodes_per_n}")
    print()
    print(f"  {'N':>4} | {'alive_rate':>11} | {'mean_life':>10} | {'mean_food':>10} | {'n_food_per_agent':>16}")
    print(f"  {'-'*4} | {'-'*11} | {'-'*10} | {'-'*10} | {'-'*16}")

    rng = np.random.default_rng(42)
    for n in args.n_agents_sweep:
        cfg = MultiAgentForagerConfig(
            rows=args.rows, cols=args.cols, n_agents=n,
            initial_food_density=args.initial_food_density,
            food_respawn_lambda=args.food_respawn_lambda, max_steps=args.max_steps,
        )
        env = MultiAgentFoodGrid(cfg)
        results = [run_one_episode(env, rng) for _ in range(args.episodes_per_n)]
        alive_rate = statistics.mean(r["alive_rate"] for r in results)
        mean_life = statistics.mean(r["mean_lifespan"] for r in results)
        mean_food = statistics.mean(r["total_food"] for r in results)
        food_per_agent = mean_food / n
        print(
            f"  {n:>4} | {alive_rate:>10.1%} | {mean_life:>10.1f} | "
            f"{mean_food:>10.1f} | {food_per_agent:>16.2f}"
        )


if __name__ == "__main__":
    main()
