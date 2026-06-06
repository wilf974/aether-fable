"""OBS Viewer 2.0 (lite) — lance l'observateur LIVE V8.

Usage:
    python scripts/launch_gui_v8.py --days 5 --ticks-per-day 1000 --device cuda
Touches : ESPACE pause · +/- jours · H Historien · E export · ↑/↓ vitesse · ESC.
"""
from __future__ import annotations

import os

# IMPORTANT : forcer un VRAI display AVANT d'importer les modules viz (qui font
# SDL_VIDEODRIVER=dummy par défaut pour les tests headless). setdefault =
# respecte un dummy déjà posé par les tests, sinon ouvre une vraie fenêtre.
os.environ.setdefault("SDL_VIDEODRIVER", "windows")

import argparse  # noqa: E402
import sys  # noqa: E402

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aetherlife.viz.live_viewer_v8 import run_live  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--regime", default="coordination_collective")
    p.add_argument("--device", default="cuda")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--ticks-per-day", type=int, default=1000)
    p.add_argument("--cell-px", type=int, default=14)
    p.add_argument("--max-frames", type=int, default=None,
                   help="borne pour smoke/test (None = illimité).")
    a = p.parse_args()
    run_live(seed=a.seed, regime=a.regime, device=a.device, days=a.days,
             ticks_per_day=a.ticks_per_day, cell_px=a.cell_px,
             max_frames=a.max_frames)


if __name__ == "__main__":
    main()
