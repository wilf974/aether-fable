"""V3.6 — benchmark 3-way MLP / LSTM / CNN+Double DQN sur env saisonnier.

Compare les 3 archis principales de MW_IA réutilisées dans AetherLife :
- MLP DQN V1.5 (recette gagnante 90%)
- DRQN LSTM V2-Y (mémoire temporelle)
- ConvDQN + Double DQN V2-W (top de MW_IA, perception spatiale + Q-target stable)

Usage :
    python scripts/v3_6_benchmark_3way.py --episodes 300 --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from mw_ia.config import ConvDQNConfig, DQNConfig, DRQNConfig

from aetherlife.agents.conv_dqn_agent import ConvDQNAgent
from aetherlife.agents.dqn_agent import DQNAgent
from aetherlife.agents.recurrent_dqn_agent import RecurrentDQNAgent
from aetherlife.training.conv_dqn_runner import run_conv_dqn_training
from aetherlife.training.dqn_runner import run_dqn_training
from aetherlife.training.recurrent_dqn_runner import run_drqn_training
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
    parser.add_argument("--season-period", type=int, default=100)
    parser.add_argument("--initial-food-density", type=float, default=0.05)
    parser.add_argument("--food-respawn-lambda", type=float, default=1.0)
    parser.add_argument("--assess-every", type=int, default=25)
    parser.add_argument("--assess-episodes", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        help="Archis à skipper : mlp lstm cnn"
    )
    args = parser.parse_args()

    seasonal = SeasonalConfig(season_period=args.season_period)
    env_cfg = SeasonalMultiAgentConfig(
        rows=args.rows, cols=args.cols, n_agents=1,
        initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda,
        max_steps=args.max_steps, seasonal=seasonal,
    )

    print(
        f"AetherLife V3.6 benchmark 3-way — episodes={args.episodes}  "
        f"device={args.device}\n  env: {env_cfg.rows}x{env_cfg.cols}  "
        f"season_period={seasonal.season_period}  "
        f"max_steps={env_cfg.max_steps}\n"
    )

    decay = args.episodes * 50
    results: dict[str, dict] = {}

    # ─── MLP DQN ──────────────────────────────────────────────────────────
    if "mlp" not in args.skip:
        print(">>> MLP DQN (V1.5 recipe)")
        env = SoloSeasonalEnv(env_cfg)
        cfg = DQNConfig(
            hidden_layers=(256, 256), epsilon_decay_steps=decay,
            lr=5e-4, batch_size=256, target_sync_steps=300, use_amp=False,
        )
        agent = DQNAgent(env, cfg, device=args.device, seed=args.seed)
        t0 = time.time()
        result = run_dqn_training(
            env, agent, n_episodes=args.episodes,
            assess_every=args.assess_every, assess_episodes=args.assess_episodes,
            checkpoint_path=Path("checkpoints/v3_6_mlp_best.pt"),
            patience=args.patience, base_seed=args.seed * 1000,
        )
        results["MLP"] = {
            "best": result.best_assessment_score, "ep": result.best_assessment_episode,
            "time": time.time() - t0,
        }
        print(
            f"  MLP best: {result.best_assessment_score:.1%}  "
            f"@ ep {result.best_assessment_episode}  ({results['MLP']['time']:.1f}s)\n"
        )

    # ─── LSTM DRQN ────────────────────────────────────────────────────────
    if "lstm" not in args.skip:
        print(">>> LSTM DRQN (V2-Y)")
        env = SoloSeasonalEnv(env_cfg)
        cfg = DRQNConfig(
            fc_hidden=128, lstm_hidden=64, sequence_length=16,
            epsilon_decay_steps=decay, lr=5e-4, batch_size=64,
            replay_capacity=2_000, target_sync_steps=300,
            train_steps_per_episode=2, max_steps_per_episode=args.max_steps,
            min_episodes_to_learn=32, use_amp=False,
        )
        agent = RecurrentDQNAgent(
            obs_dim=env.n_states, n_actions=env.n_actions,
            cfg=cfg, device=args.device, seed=args.seed,
        )
        t0 = time.time()
        result = run_drqn_training(
            env, agent, n_episodes=args.episodes,
            assess_every=args.assess_every, assess_episodes=args.assess_episodes,
            checkpoint_path=Path("checkpoints/v3_6_lstm_best.pt"),
            patience=args.patience, base_seed=args.seed * 1000,
        )
        results["LSTM"] = {
            "best": result.best_assessment_score, "ep": result.best_assessment_episode,
            "time": time.time() - t0,
        }
        print(
            f"  LSTM best: {result.best_assessment_score:.1%}  "
            f"@ ep {result.best_assessment_episode}  ({results['LSTM']['time']:.1f}s)\n"
        )

    # ─── CNN + Double DQN ─────────────────────────────────────────────────
    if "cnn" not in args.skip:
        print(">>> ConvDQN + Double DQN (V2-W TOP)")
        env = SoloSeasonalEnv(env_cfg)
        cfg = ConvDQNConfig(
            conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
            epsilon_decay_steps=decay, lr=5e-4, batch_size=256,
            target_sync_steps=300, train_every=4, use_amp=False,
            double_dqn=True,  # V2-W
        )
        in_ch, R, C = env.obs_2d_shape
        agent = ConvDQNAgent(
            in_channels=in_ch, rows=R, cols=C, n_actions=env.n_actions,
            cfg=cfg, device=args.device, seed=args.seed,
        )
        t0 = time.time()
        result = run_conv_dqn_training(
            env, agent, n_episodes=args.episodes,
            assess_every=args.assess_every, assess_episodes=args.assess_episodes,
            checkpoint_path=Path("checkpoints/v3_6_cnn_ddqn_best.pt"),
            patience=args.patience, base_seed=args.seed * 1000,
        )
        results["CNN+DDQN"] = {
            "best": result.best_assessment_score, "ep": result.best_assessment_episode,
            "time": time.time() - t0,
        }
        print(
            f"  CNN+DDQN best: {result.best_assessment_score:.1%}  "
            f"@ ep {result.best_assessment_episode}  ({results['CNN+DDQN']['time']:.1f}s)\n"
        )

    # ─── comparaison ──────────────────────────────────────────────────────
    print("=== RESULTS ===")
    print(f"  {'archi':<10} | {'best':>8} | {'@ ep':>5} | {'time':>7}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*5} | {'-'*7}")
    for name, r in results.items():
        print(
            f"  {name:<10} | {r['best']:>7.1%} | {r['ep']:>5} | {r['time']:>6.1f}s"
        )

    if len(results) >= 2:
        best_archi = max(results.items(), key=lambda kv: kv[1]["best"])
        print(f"\n  WINNER : {best_archi[0]} ({best_archi[1]['best']:.1%})")


if __name__ == "__main__":
    main()
