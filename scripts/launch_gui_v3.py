"""Lance la GUI V3 saisonnière multi-agent AetherLife.

Usage :
    python scripts/launch_gui_v3.py
    python scripts/launch_gui_v3.py --n-agents 8 --season-period 80
"""
from __future__ import annotations

import argparse

from aetherlife.viz.pygame_viewer_v3 import run_gui_v3
from aetherlife.world.seasonal_grid import SeasonalConfig, SeasonalMultiAgentConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-energy", type=float, default=100.0)
    parser.add_argument("--start-energy", type=float, default=50.0)
    parser.add_argument("--metabolism", type=float, default=1.0)
    parser.add_argument("--food-value", type=float, default=20.0)
    parser.add_argument("--initial-food-density", type=float, default=0.05)
    parser.add_argument("--food-respawn-lambda", type=float, default=1.0)
    parser.add_argument("--season-period", type=int, default=200)
    parser.add_argument("--spring-factor", type=float, default=1.5)
    parser.add_argument("--summer-factor", type=float, default=1.0)
    parser.add_argument("--autumn-factor", type=float, default=1.2)
    parser.add_argument("--winter-factor", type=float, default=0.3)
    parser.add_argument("--temp-min", type=float, default=-10.0)
    parser.add_argument("--temp-max", type=float, default=30.0)
    parser.add_argument("--cell-px", type=int, default=18)
    parser.add_argument("--tick-delay-ms", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seasonal = SeasonalConfig(
        season_period=args.season_period,
        spring_lambda_factor=args.spring_factor,
        summer_lambda_factor=args.summer_factor,
        autumn_lambda_factor=args.autumn_factor,
        winter_lambda_factor=args.winter_factor,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
    )
    cfg = SeasonalMultiAgentConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        max_energy=args.max_energy, start_energy=args.start_energy,
        metabolism=args.metabolism, food_value=args.food_value,
        initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda,
        max_steps=args.max_steps, seasonal=seasonal,
    )
    run_gui_v3(cfg, cell_px=args.cell_px, tick_delay_ms=args.tick_delay_ms, seed=args.seed)


if __name__ == "__main__":
    main()
