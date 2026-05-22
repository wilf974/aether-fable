"""Smoke baseline V1 — compare RandomAgent et GreedyAgent sur N épisodes.

Usage :
    python scripts/run_baseline.py --episodes 100
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

from aetherlife.agents.greedy_agent import GreedyAgent
from aetherlife.agents.random_agent import RandomAgent
from aetherlife.config import FoodGridConfig
from aetherlife.env.single_agent_env import SoloForagerEnv


@dataclass
class EpisodeStats:
    survived: bool
    lifespan: int
    total_reward: float
    food_eaten: int


def run_episode(env: SoloForagerEnv, agent: object, seed: int) -> EpisodeStats:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    food_eaten = 0
    step = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.act(obs)  # type: ignore[attr-defined]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info.get("ate"):
            food_eaten += 1
        step += 1
    return EpisodeStats(
        survived=truncated,
        lifespan=step,
        total_reward=total_reward,
        food_eaten=food_eaten,
    )


def summarize(name: str, stats: list[EpisodeStats]) -> dict[str, float]:
    survival_rate = sum(1 for s in stats if s.survived) / len(stats)
    mean_lifespan = statistics.mean(s.lifespan for s in stats)
    mean_reward = statistics.mean(s.total_reward for s in stats)
    mean_food = statistics.mean(s.food_eaten for s in stats)
    print(
        f"  {name:14s}: survival={survival_rate:6.1%}  "
        f"lifespan={mean_lifespan:7.1f}  "
        f"reward={mean_reward:+8.1f}  food={mean_food:5.1f}"
    )
    return {
        "survival_rate": survival_rate,
        "mean_lifespan": mean_lifespan,
        "mean_reward": mean_reward,
        "mean_food": mean_food,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--start-energy", type=float, default=50.0)
    parser.add_argument("--max-energy", type=float, default=100.0)
    parser.add_argument("--metabolism", type=float, default=1.0)
    parser.add_argument("--food-value", type=float, default=20.0)
    parser.add_argument("--initial-food-density", type=float, default=0.05)
    parser.add_argument("--food-respawn-lambda", type=float, default=0.5)
    args = parser.parse_args()

    cfg = FoodGridConfig(
        rows=args.rows,
        cols=args.cols,
        max_energy=args.max_energy,
        start_energy=args.start_energy,
        metabolism=args.metabolism,
        food_value=args.food_value,
        initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda,
        max_steps=args.max_steps,
        start_position=(args.rows // 2, args.cols // 2),
    )

    print(
        f"AetherLife V1 baseline — episodes={args.episodes}  "
        f"grid={cfg.rows}x{cfg.cols}  max_steps={cfg.max_steps}"
    )
    print(
        f"  config: metabolism={cfg.metabolism} food_value={cfg.food_value} "
        f"density={cfg.initial_food_density} respawn_lambda={cfg.food_respawn_lambda}"
    )
    print()

    env = SoloForagerEnv(cfg)
    random_agent = RandomAgent(n_actions=4, seed=0)
    greedy_agent = GreedyAgent(rows=cfg.rows, cols=cfg.cols, seed=0)

    rand_stats = [run_episode(env, random_agent, seed=ep) for ep in range(args.episodes)]
    summarize("RandomAgent", rand_stats)

    greedy_stats = [run_episode(env, greedy_agent, seed=ep) for ep in range(args.episodes)]
    summarize("GreedyAgent", greedy_stats)


if __name__ == "__main__":
    main()
