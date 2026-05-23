"""Lance la GUI V2 multi-agent AetherLife.

Usage :
    python scripts/launch_gui_v2.py
    python scripts/launch_gui_v2.py --n-agents 8 --rows 24 --cols 24 --cell-px 24
    python scripts/launch_gui_v2.py --idqn-checkpoint checkpoints/ma_idqn_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aetherlife.viz.pygame_viewer_v2 import run_gui_v2
from aetherlife.world.multi_agent_grid import MultiAgentForagerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-energy", type=float, default=100.0)
    parser.add_argument("--start-energy", type=float, default=50.0)
    parser.add_argument("--metabolism", type=float, default=1.0)
    parser.add_argument("--food-value", type=float, default=20.0)
    parser.add_argument("--initial-food-density", type=float, default=0.05)
    parser.add_argument("--food-respawn-lambda", type=float, default=1.0)
    parser.add_argument("--cell-px", type=int, default=18)
    parser.add_argument("--tick-delay-ms", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--idqn-checkpoint", type=str, default=None,
        help="Chemin du checkpoint IDQN à charger (ajoute slot IDQN dans le switch).",
    )
    args = parser.parse_args()

    cfg = MultiAgentForagerConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        max_energy=args.max_energy, start_energy=args.start_energy,
        metabolism=args.metabolism, food_value=args.food_value,
        initial_food_density=args.initial_food_density,
        food_respawn_lambda=args.food_respawn_lambda, max_steps=args.max_steps,
    )
    ckpt = Path(args.idqn_checkpoint) if args.idqn_checkpoint else None
    run_gui_v2(
        cfg, cell_px=args.cell_px, tick_delay_ms=args.tick_delay_ms,
        seed=args.seed, idqn_checkpoint=ckpt,
    )


if __name__ == "__main__":
    main()
