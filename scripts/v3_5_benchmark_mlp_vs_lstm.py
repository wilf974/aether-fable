"""V3.5 — benchmark MLP vs LSTM sur env saisonnier single-agent.

Test H3 : LSTM > MLP face à des régimes non-stationnaires (saisons).
Entraîne les deux archis sur le même env saisonnier, compare assessment.

Usage :
    python scripts/v3_5_benchmark_mlp_vs_lstm.py --episodes 300 --device cuda
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mw_ia.config import DQNConfig, DRQNConfig

from aetherlife.agents.dqn_agent import DQNAgent
from aetherlife.agents.recurrent_dqn_agent import RecurrentDQNAgent
from aetherlife.config import FoodGridConfig
from aetherlife.training.dqn_runner import run_dqn_training
from aetherlife.training.recurrent_dqn_runner import run_drqn_training
from aetherlife.world.food_grid import FoodGrid
from aetherlife.world.seasonal_grid import SeasonalConfig, SeasonalMultiAgentConfig
from aetherlife.world.seasonal_solo_env import SoloSeasonalEnv


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
    parser.add_argument("--food-respawn-lambda", type=float, default=1.0)
    parser.add_argument("--season-period", type=int, default=100)
    parser.add_argument("--winter-factor", type=float, default=0.3)
    parser.add_argument("--summer-factor", type=float, default=1.0)
    parser.add_argument("--spring-factor", type=float, default=1.5)
    parser.add_argument("--autumn-factor", type=float, default=1.2)
    parser.add_argument("--assess-every", type=int, default=25)
    parser.add_argument("--assess-episodes", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    seasonal = SeasonalConfig(
        season_period=args.season_period,
        spring_lambda_factor=args.spring_factor,
        summer_lambda_factor=args.summer_factor,
        autumn_lambda_factor=args.autumn_factor,
        winter_lambda_factor=args.winter_factor,
    )
    env_cfg = SeasonalMultiAgentConfig(
        rows=args.rows, cols=args.cols, n_agents=1,
        max_energy=args.max_energy, start_energy=args.start_energy,
        metabolism=args.metabolism, food_value=args.food_value,
        initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda,
        max_steps=args.max_steps, seasonal=seasonal,
    )

    print(
        f"AetherLife V3.5 benchmark MLP vs LSTM — episodes={args.episodes}  "
        f"device={args.device}\n  env: {env_cfg.rows}x{env_cfg.cols}  "
        f"season_period={seasonal.season_period}  "
        f"factors=spring/summer/autumn/winter "
        f"{seasonal.spring_lambda_factor}/{seasonal.summer_lambda_factor}/"
        f"{seasonal.autumn_lambda_factor}/{seasonal.winter_lambda_factor}\n"
    )

    # ─── MLP DQN (recette V1.5 gagnante) ──────────────────────────────────
    print(">>> MLP DQN (V1.5 recipe sur env saisonnier)")
    env_mlp = SoloSeasonalEnv(env_cfg)
    dqn_cfg = DQNConfig(
        hidden_layers=(256, 256), episodes=args.episodes,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=args.episodes * 50,
        gamma=0.99, lr=5e-4, batch_size=256,
        replay_capacity=100_000, target_sync_steps=300, use_amp=False,
    )
    agent_mlp = DQNAgent(env_mlp, dqn_cfg, device=args.device, seed=args.seed)
    t0 = time.time()
    result_mlp = run_dqn_training(
        env_mlp, agent_mlp,
        n_episodes=args.episodes,
        assess_every=args.assess_every, assess_episodes=args.assess_episodes,
        checkpoint_path=Path("checkpoints/mlp_seasonal_best.pt"),
        patience=args.patience, base_seed=args.seed * 1000,
    )
    t_mlp = time.time() - t0
    print(
        f"  MLP best assessment: {result_mlp.best_assessment_score:.1%}  "
        f"@ ep {result_mlp.best_assessment_episode}  ({t_mlp:.1f}s)\n"
    )

    # ─── LSTM DRQN ────────────────────────────────────────────────────────
    print(">>> LSTM DRQN")
    env_lstm = SoloSeasonalEnv(env_cfg)
    drqn_cfg = DRQNConfig(
        fc_hidden=128, lstm_hidden=64, sequence_length=16,
        episodes=args.episodes, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=args.episodes * 50,
        gamma=0.99, lr=5e-4, batch_size=64,
        replay_capacity=2_000, target_sync_steps=300,
        train_steps_per_episode=2, max_steps_per_episode=args.max_steps,
        min_episodes_to_learn=32, use_amp=False,
    )
    agent_lstm = RecurrentDQNAgent(
        obs_dim=env_lstm.n_states, n_actions=env_lstm.n_actions,
        cfg=drqn_cfg, device=args.device, seed=args.seed,
    )
    t0 = time.time()
    result_lstm = run_drqn_training(
        env_lstm, agent_lstm,
        n_episodes=args.episodes,
        assess_every=args.assess_every, assess_episodes=args.assess_episodes,
        checkpoint_path=Path("checkpoints/lstm_seasonal_best.pt"),
        patience=args.patience, base_seed=args.seed * 1000,
    )
    t_lstm = time.time() - t0
    print(
        f"  LSTM best assessment: {result_lstm.best_assessment_score:.1%}  "
        f"@ ep {result_lstm.best_assessment_episode}  ({t_lstm:.1f}s)\n"
    )

    # ─── comparaison ──────────────────────────────────────────────────────
    print("=== RESULT ===")
    print(f"  MLP  best survival rate: {result_mlp.best_assessment_score:.1%}")
    print(f"  LSTM best survival rate: {result_lstm.best_assessment_score:.1%}")
    diff = result_lstm.best_assessment_score - result_mlp.best_assessment_score
    print(f"  delta LSTM vs MLP      : {diff:+.1%}")
    if abs(diff) < 0.05:
        print("  -> pas de difference significative (Delta < 5 pp)")
    elif diff > 0:
        print(f"  -> LSTM bat MLP de {diff*100:.1f} pp * (H3 supporte)")
    else:
        print(f"  -> MLP bat LSTM de {-diff*100:.1f} pp (H3 non confirmee)")


if __name__ == "__main__":
    main()
