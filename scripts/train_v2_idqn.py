"""Entraîne un IDQN shared-weights sur AetherLife V2 Multi-Agent.

Usage :
    python scripts/train_v2_idqn.py --n-agents 16 --episodes 300 --device cuda
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mw_ia.config import DQNConfig

from aetherlife.agents.independent_dqn import IndependentDQNAgent
from aetherlife.training.multi_agent_runner import (
    MAAssessmentMetric,
    MAEpisodeMetric,
    run_ma_training,
)
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-energy", type=float, default=100.0)
    parser.add_argument("--start-energy", type=float, default=50.0)
    parser.add_argument("--metabolism", type=float, default=1.0)
    parser.add_argument("--food-value", type=float, default=20.0)
    parser.add_argument("--initial-food-density", type=float, default=0.05)
    parser.add_argument("--food-respawn-lambda", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--epsilon-decay-steps", type=int, default=40_000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--target-sync-steps", type=int, default=300)
    parser.add_argument("--assess-every", type=int, default=25)
    parser.add_argument("--assess-episodes", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ma_idqn_best.pt")
    args = parser.parse_args()

    env_cfg = MultiAgentForagerConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        max_energy=args.max_energy, start_energy=args.start_energy,
        metabolism=args.metabolism, food_value=args.food_value,
        initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda, max_steps=args.max_steps,
    )
    dqn_cfg = DQNConfig(
        hidden_layers=tuple(args.hidden),
        episodes=args.episodes,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=args.epsilon_decay_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_sync_steps=args.target_sync_steps,
        use_amp=False,
    )

    env = MultiAgentFoodGrid(env_cfg)
    agent = IndependentDQNAgent(env, dqn_cfg, device=args.device, seed=args.seed)

    print(
        f"AetherLife V2 IDQN — episodes={args.episodes}  n_agents={args.n_agents}  "
        f"device={args.device}\n"
        f"  env: {env_cfg.rows}x{env_cfg.cols} density={env_cfg.initial_food_density} "
        f"respawn={env_cfg.food_respawn_lambda} max_steps={env_cfg.max_steps}\n"
    )

    def log_episode(m: MAEpisodeMetric) -> None:
        if (m.episode + 1) % 10 == 0 or m.episode == 0:
            loss_str = f"{m.last_loss:.4f}" if m.last_loss is not None else "n/a"
            print(
                f"  ep {m.episode+1:4d}  alive={m.n_alive_final}/{env_cfg.n_agents}  "
                f"life={m.mean_lifespan:5.1f}  food={m.total_food_eaten:4d}  "
                f"R/agent={m.total_reward/max(1, env_cfg.n_agents):+6.1f}  "
                f"eps={m.epsilon:.3f}  loss={loss_str}"
            )

    def log_assess(m: MAAssessmentMetric, improved: bool) -> None:
        mark = "* NEW BEST" if improved else "  "
        print(
            f"  >>> assess @ step {m.train_episode:7d}: "
            f"alive_rate={m.mean_alive_rate:5.1%}  "
            f"life={m.mean_lifespan:6.1f}  food={m.mean_total_food:5.1f}  {mark}"
        )

    t0 = time.time()
    result = run_ma_training(
        env, agent,
        n_episodes=args.episodes,
        assess_every=args.assess_every,
        assess_episodes=args.assess_episodes,
        checkpoint_path=Path(args.checkpoint),
        patience=args.patience,
        base_seed=args.seed * 1000,
        on_episode_end=log_episode,
        on_assess=log_assess,
    )
    elapsed = time.time() - t0

    last_50 = result.train_metrics[-50:]
    mean_life_last = (
        statistics.mean(m.mean_lifespan for m in last_50) if last_50 else 0.0
    )

    print(f"\n--- V2 training finished ({elapsed:.1f}s) ---")
    print(f"  episodes ran             : {result.final_episode + 1}")
    print(f"  stopped early            : {result.stopped_early}")
    print(f"  mean lifespan (last 50)  : {mean_life_last:.1f}")
    print(f"  best alive rate          : {result.best_assessment_score:.1%}")
    print(f"  best @ episode           : {result.best_assessment_episode}")
    print(f"  checkpoint               : {args.checkpoint}")


if __name__ == "__main__":
    main()
