"""V3.7 — benchmark multi-agent saisonnier complexe : MLP IDQN vs CNN+DDQN.

Objectif : prouver que CNN+DDQN dépasse MLP quand la perception spatiale
devient nécessaire (env multi-agent + saisons contraignantes + density basse).

Setup par défaut :
  - 32×32 grille
  - N=16 agents
  - density 0.02 (basse)
  - season_period 80 ticks
  - winter_factor=0.2 (rare en hiver), spring_factor=2.0 (boom printanier)
  - assessment held-out (seeds eval séparés du training)

Usage :
    python scripts/v3_7_benchmark_ma_seasonal_complex.py --episodes 400 --device cuda
    python scripts/v3_7_benchmark_ma_seasonal_complex.py --n-seeds 3 --episodes 400
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mw_ia.config import ConvDQNConfig, DQNConfig

from aetherlife.agents.independent_conv_dqn import IndependentConvDQNAgent
from aetherlife.agents.independent_dqn import IndependentDQNAgent
from aetherlife.training.multi_agent_conv_runner import (
    MAConvAssessmentMetric,
    MAConvEpisodeMetric,
    run_ma_conv_training,
)
from aetherlife.training.multi_agent_runner import (
    MAAssessmentMetric,
    MAEpisodeMetric,
    run_ma_training,
)
from aetherlife.world.multi_agent_grid import MultiAgentForagerConfig
from aetherlife.world.multi_agent_grid import MultiAgentFoodGrid
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


def run_mlp_seed(args, seasonal, seed: int) -> tuple[float, int, float]:
    """Entraîne IDQN MLP sur env NON-saisonnier (V2 IDQN n'utilise pas les saisons).
    NB : V2 IndependentDQNAgent attend env.n_states (3*H*W + 1), pas env saisonnier.
    On utilise donc le MultiAgentFoodGrid V2 (sans saisons) comme baseline MLP.
    """
    env_cfg = MultiAgentForagerConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda,
        max_steps=args.max_steps,
    )
    env = MultiAgentFoodGrid(env_cfg)
    cfg = DQNConfig(
        hidden_layers=(256, 256), epsilon_decay_steps=args.episodes * 50,
        lr=5e-4, batch_size=256, target_sync_steps=300, use_amp=False,
    )
    agent = IndependentDQNAgent(env, cfg, device=args.device, seed=seed)
    t0 = time.time()
    result = run_ma_training(
        env, agent, n_episodes=args.episodes,
        assess_every=args.assess_every, assess_episodes=args.assess_episodes,
        checkpoint_path=Path(f"checkpoints/v3_7_mlp_seed{seed}_best.pt"),
        patience=args.patience, base_seed=seed * 1000,
    )
    return result.best_assessment_score, result.best_assessment_episode, time.time() - t0


def run_cnn_seed(args, seasonal, seed: int) -> tuple[float, int, float]:
    """Entraîne IndependentConvDQNAgent (V2-W shared-weights) sur env saisonnier 2D."""
    env_cfg = SeasonalMultiAgentConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda,
        max_steps=args.max_steps, seasonal=seasonal,
    )
    env = SeasonalMultiAgentFoodGrid(env_cfg)
    cfg = ConvDQNConfig(
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
        epsilon_decay_steps=args.episodes * 50, lr=5e-4, batch_size=256,
        target_sync_steps=300, train_every=4, use_amp=False,
        double_dqn=True,
    )
    agent = IndependentConvDQNAgent(env, cfg, device=args.device, seed=seed)
    t0 = time.time()
    result = run_ma_conv_training(
        env, agent, n_episodes=args.episodes,
        assess_every=args.assess_every, assess_episodes=args.assess_episodes,
        checkpoint_path=Path(f"checkpoints/v3_7_cnn_seed{seed}_best.pt"),
        patience=args.patience, base_seed=seed * 1000,
    )
    return result.best_assessment_score, result.best_assessment_episode, time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--season-period", type=int, default=80)
    parser.add_argument("--initial-food-density", type=float, default=0.02)
    parser.add_argument("--food-respawn-lambda", type=float, default=0.8)
    parser.add_argument("--winter-factor", type=float, default=0.2)
    parser.add_argument("--summer-factor", type=float, default=1.0)
    parser.add_argument("--spring-factor", type=float, default=2.0)
    parser.add_argument("--autumn-factor", type=float, default=1.2)
    parser.add_argument("--assess-every", type=int, default=25)
    parser.add_argument("--assess-episodes", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        help="Archis a skipper : mlp cnn"
    )
    args = parser.parse_args()

    seasonal = SeasonalConfig(
        season_period=args.season_period,
        spring_lambda_factor=args.spring_factor,
        summer_lambda_factor=args.summer_factor,
        autumn_lambda_factor=args.autumn_factor,
        winter_lambda_factor=args.winter_factor,
    )

    print(
        f"AetherLife V3.7 benchmark MA complex - episodes={args.episodes}  "
        f"device={args.device}\n  env: {args.rows}x{args.cols}  "
        f"N={args.n_agents}  density={args.initial_food_density}  "
        f"respawn={args.food_respawn_lambda}  max_steps={args.max_steps}\n  "
        f"saisons: period={seasonal.season_period}  "
        f"spring/summer/autumn/winter = "
        f"{seasonal.spring_lambda_factor}/{seasonal.summer_lambda_factor}/"
        f"{seasonal.autumn_lambda_factor}/{seasonal.winter_lambda_factor}\n  "
        f"seeds={args.n_seeds}  assess_every={args.assess_every} ep, "
        f"held-out seeds [100000..)\n"
    )

    mlp_results: list[tuple[float, int, float]] = []
    cnn_results: list[tuple[float, int, float]] = []

    for seed in range(args.n_seeds):
        if "mlp" not in args.skip:
            print(f">>> seed {seed} MLP IDQN (V2 baseline)")
            r = run_mlp_seed(args, seasonal, seed)
            mlp_results.append(r)
            print(
                f"  MLP seed {seed}: best alive_rate={r[0]:.1%} @ ep {r[1]}  ({r[2]:.1f}s)"
            )

        if "cnn" not in args.skip:
            print(f">>> seed {seed} CNN+DDQN IDQN (V3.7 spatial)")
            r = run_cnn_seed(args, seasonal, seed)
            cnn_results.append(r)
            print(
                f"  CNN seed {seed}: best alive_rate={r[0]:.1%} @ ep {r[1]}  ({r[2]:.1f}s)"
            )
        print()

    print("=== RESULTS V3.7 ===")
    print(f"  {'archi':<10} | {'best mean':>10} | {'best std':>9} | {'time mean':>10}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*9} | {'-'*10}")
    if mlp_results:
        bests = [r[0] for r in mlp_results]
        times = [r[2] for r in mlp_results]
        m = statistics.mean(bests)
        s = statistics.pstdev(bests) if len(bests) > 1 else 0.0
        t = statistics.mean(times)
        print(f"  {'MLP':<10} | {m:>9.1%} | {s:>8.1%} | {t:>9.1f}s")
    if cnn_results:
        bests = [r[0] for r in cnn_results]
        times = [r[2] for r in cnn_results]
        m = statistics.mean(bests)
        s = statistics.pstdev(bests) if len(bests) > 1 else 0.0
        t = statistics.mean(times)
        print(f"  {'CNN+DDQN':<10} | {m:>9.1%} | {s:>8.1%} | {t:>9.1f}s")

    if mlp_results and cnn_results:
        mlp_mean = statistics.mean(r[0] for r in mlp_results)
        cnn_mean = statistics.mean(r[0] for r in cnn_results)
        diff = cnn_mean - mlp_mean
        print(f"\n  CNN+DDQN vs MLP delta: {diff:+.1%}")
        if abs(diff) < 0.03:
            print("  -> pas de difference significative (Delta < 3 pp)")
        elif diff > 0:
            print(f"  -> CNN+DDQN bat MLP de {diff*100:.1f} pp * (perception spatiale justifie)")
        else:
            print(f"  -> MLP bat CNN+DDQN de {-diff*100:.1f} pp")


if __name__ == "__main__":
    main()
