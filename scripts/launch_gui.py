"""Lance la GUI live AetherLife V1/V1.5.

Usage :
    python scripts/launch_gui.py
    python scripts/launch_gui.py --rows 24 --cols 24 --cell-px 24
    python scripts/launch_gui.py --dqn-checkpoint checkpoints/dqn_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aetherlife.config import FoodGridConfig
from aetherlife.viz.pygame_viewer import run_gui


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-energy", type=float, default=100.0)
    parser.add_argument("--start-energy", type=float, default=50.0)
    parser.add_argument("--metabolism", type=float, default=1.0)
    parser.add_argument("--food-value", type=float, default=20.0)
    parser.add_argument("--initial-food-density", type=float, default=0.05)
    parser.add_argument("--food-respawn-lambda", type=float, default=0.5)
    parser.add_argument("--cell-px", type=int, default=32)
    parser.add_argument("--tick-delay-ms", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dqn-checkpoint", type=str, default=None,
        help="Chemin du checkpoint DQN à charger (ajoute slot DQN dans le switch).",
    )
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
    ckpt = Path(args.dqn_checkpoint) if args.dqn_checkpoint else None
    run_gui(
        cfg, cell_px=args.cell_px, tick_delay_ms=args.tick_delay_ms,
        seed=args.seed, dqn_checkpoint=ckpt,
    )


if __name__ == "__main__":
    main()
