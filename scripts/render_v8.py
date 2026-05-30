"""CLI — rend un events.jsonl V8 en clip PNG/GIF/MP4.

Usage:
    python scripts/render_v8.py --events results/seed25/events.jsonl \\
        --out clips/seed25.mp4 --fps 30 [--from-tick 0 --to-tick 16000] \\
        [--focus-lineage 12] [--cell-px 16]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aetherlife.viz.pygame_viewer_v8 import render_events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--meta", default=None, help="défaut: meta.json à côté de --events")
    p.add_argument("--out", required=True, help="fichier .mp4/.gif, ou dossier si .png")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--from-tick", type=int, default=0)
    p.add_argument("--to-tick", type=int, default=None)
    p.add_argument("--focus-lineage", type=int, default=None)
    p.add_argument("--cell-px", type=int, default=16)
    a = p.parse_args()

    meta = a.meta or os.path.join(os.path.dirname(a.events), "meta.json")
    ext = os.path.splitext(a.out)[1].lower().lstrip(".")
    fmt = ext if ext in ("png", "gif", "mp4") else "mp4"
    res = render_events(
        a.events, meta, a.out, fmt=fmt, fps=a.fps,
        from_tick=a.from_tick, to_tick=a.to_tick,
        focus_lineage=a.focus_lineage, cell_px=a.cell_px,
    )
    print(f"WROTE {res}")


if __name__ == "__main__":
    main()
