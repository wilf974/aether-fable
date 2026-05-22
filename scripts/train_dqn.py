"""Entraîne un DQN sur AetherLife V1 Solo Forager + assessment greedy + best-checkpoint.

Usage :
    python scripts/train_dqn.py --episodes 300 --device cuda
    python scripts/train_dqn.py --episodes 100 --device cpu --hidden 64 64
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mw_ia.config import DQNConfig

from aetherlife.agents.dqn_agent import DQNAgent
from aetherlife.config import FoodGridConfig
from aetherlife.training.dqn_runner import (
    AssessmentMetric,
    EpisodeMetric,
    run_dqn_training,
)
from aetherlife.world.food_grid import FoodGrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-energy", type=float, default=100.0)
    parser.add_argument("--start-energy", type=float, default=50.0)
    parser.add_argument("--metabolism", type=float, default=1.0)
    parser.add_argument("--food-value", type=float, default=20.0)
    parser.add_argument("--initial-food-density", type=float, default=0.05)
    parser.add_argument("--food-respawn-lambda", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-sync-steps", type=int, default=1_000)
    parser.add_argument("--assess-every", type=int, default=25)
    parser.add_argument("--assess-episodes", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dqn_best.pt")
    args = parser.parse_args()

    env_cfg = FoodGridConfig(
        rows=args.rows, cols=args.cols, max_energy=args.max_energy,
        start_energy=args.start_energy, metabolism=args.metabolism,
        food_value=args.food_value, initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda, max_steps=args.max_steps,
        start_position=(args.rows // 2, args.cols // 2),
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

    env = FoodGrid(env_cfg)
    agent = DQNAgent(env, dqn_cfg, device=args.device, seed=args.seed)

    print(
        f"AetherLife V1.5 DQN training — episodes={args.episodes}  device={args.device}\n"
        f"  env: {env_cfg.rows}x{env_cfg.cols} food_density={env_cfg.initial_food_density} "
        f"respawn_lambda={env_cfg.food_respawn_lambda} max_steps={env_cfg.max_steps}\n"
        f"  dqn: hidden={dqn_cfg.hidden_layers} lr={dqn_cfg.lr} "
        f"epsilon_decay_steps={dqn_cfg.epsilon_decay_steps}\n"
        f"  assess_every={args.assess_every} ep, n={args.assess_episodes} ep, patience={args.patience}\n"
    )

    def log_episode(m: EpisodeMetric) -> None:
        if (m.episode + 1) % 10 == 0 or m.episode == 0:
            loss_str = f"{m.last_loss:.4f}" if m.last_loss is not None else "n/a"
            print(
                f"  ep {m.episode+1:4d}  reward={m.total_reward:+8.1f}  "
                f"life={m.lifespan:4d}  food={m.food_eaten:3d}  "
                f"survived={m.survived}  eps={m.epsilon:.3f}  loss={loss_str}"
            )

    def log_assess(m: AssessmentMetric, improved: bool) -> None:
        mark = "* NEW BEST" if improved else "  "
        print(
            f"  >>> assess @ step {m.train_episode:6d}: survival={m.survival_rate:6.1%}  "
            f"life={m.mean_lifespan:7.1f}  food={m.mean_food:5.1f}  {mark}"
        )

    t0 = time.time()
    result = run_dqn_training(
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

    n_survived = sum(1 for m in result.train_metrics if m.survived)
    last_50 = result.train_metrics[-50:]
    mean_reward_last = (
        statistics.mean(m.total_reward for m in last_50) if last_50 else 0.0
    )

    print(f"\n--- training finished ({elapsed:.1f}s) ---")
    print(f"  episodes ran             : {result.final_episode + 1}")
    print(f"  stopped early            : {result.stopped_early}")
    print(f"  training survival rate   : {n_survived / max(1, len(result.train_metrics)):.1%}")
    print(f"  mean reward (last 50 ep) : {mean_reward_last:+.1f}")
    print(f"  best assessment score    : {result.best_assessment_score:.1%}")
    print(f"  best @ episode           : {result.best_assessment_episode}")
    print(f"  checkpoint saved at      : {args.checkpoint}")


if __name__ == "__main__":
    main()
