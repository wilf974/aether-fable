import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from aetherlife.viz.pygame_viewer_v8 import _draw_frame
from aetherlife.viz.v8_replay import lineage_color, token_color

META = {"rows": 6, "cols": 6, "n_tokens": 4}
CELL = 10


def test_frame_dimensions_include_hud():
    ev = {"t": 1, "agents": []}
    surf = _draw_frame(ev, META, cell_px=CELL)
    # largeur = cols*cell ; hauteur = rows*cell + HUD
    assert surf.get_width() == META["cols"] * CELL
    assert surf.get_height() > META["rows"] * CELL


def test_agent_cell_tinted_by_lineage():
    ev = {"t": 1, "agents": [{"id": 0, "lin": 12, "r": 2, "c": 3}]}
    surf = _draw_frame(ev, META, cell_px=CELL)
    # centre de la cellule (c=3,r=2) : x=3*10+5=35, y=2*10+5=25
    px = surf.get_at((35, 25))[:3]
    assert tuple(px) == lineage_color(12)


def test_vocalize_halo_uses_token_color():
    ev = {
        "t": 1,
        "agents": [{"id": 7, "lin": 5, "r": 2, "c": 3}],
        "vocal": {"7": 1},
    }
    surf = _draw_frame(ev, META, cell_px=CELL)
    # halo au sommet de la cellule : cx=3*10+5=35, cy=2*10+radius
    radius = max(2, CELL // 4)
    px = surf.get_at((35, 2 * CELL + radius))[:3]
    assert tuple(px) == token_color(1)


import json
from aetherlife.viz.pygame_viewer_v8 import render_events


def _write_run(tmp_path):
    meta = {"rows": 6, "cols": 6, "n_tokens": 4}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    events = [
        {"t": 10, "n_alive": 1, "n_lin": 1, "season": 0,
         "agents": [{"id": 0, "lin": 3, "r": 1, "c": 1}], "vocal": {}, "spots": []},
        {"t": 20, "n_alive": 1, "n_lin": 1, "season": 0,
         "agents": [{"id": 0, "lin": 3, "r": 2, "c": 2}], "vocal": {"0": 2},
         "spots": [{"r": 4, "c": 4, "n": 2}]},
        {"t": 30, "n_alive": 1, "n_lin": 1, "season": 1,
         "agents": [{"id": 0, "lin": 3, "r": 3, "c": 3}], "vocal": {}, "spots": []},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return str(tmp_path / "events.jsonl"), str(tmp_path / "meta.json")


def test_render_png_writes_one_frame_per_event(tmp_path):
    events, meta = _write_run(tmp_path)
    out_dir = str(tmp_path / "frames")
    paths = render_events(events, meta, out_dir, fmt="png", cell_px=8)
    assert len(paths) == 3
    assert all(os.path.getsize(p) > 0 for p in paths)


def test_render_png_respects_tick_range(tmp_path):
    events, meta = _write_run(tmp_path)
    out_dir = str(tmp_path / "frames2")
    paths = render_events(events, meta, out_dir, fmt="png", cell_px=8,
                          from_tick=15, to_tick=25)
    assert len(paths) == 1  # seul t=20 dans [15,25]


def test_render_mp4_produces_nonempty_file(tmp_path):
    events, meta = _write_run(tmp_path)
    out = str(tmp_path / "clip.mp4")
    res = render_events(events, meta, out, fmt="mp4", fps=5, cell_px=8)
    assert res == out
    assert os.path.getsize(out) > 0


def test_focus_lineage_dims_other_lineages():
    ev = {"t": 1, "agents": [
        {"id": 0, "lin": 5, "r": 1, "c": 1},
        {"id": 1, "lin": 9, "r": 3, "c": 3},
    ]}
    surf = _draw_frame(ev, META, cell_px=CELL, focus_lineage=5)
    # agent lin=9 (hors focus) en (r=3,c=3) : centre x=3*10+5=35, y=3*10+5=35
    px = surf.get_at((35, 35))[:3]
    base = lineage_color(9)
    assert tuple(px) == (base[0] // 3, base[1] // 3, base[2] // 3)


import subprocess
import sys


def test_render_v8_cli_end_to_end(tmp_path):
    events, meta = _write_run(tmp_path)
    out = str(tmp_path / "cli_clip.mp4")
    r = subprocess.run(
        [sys.executable, "scripts/render_v8.py", "--events", events,
         "--out", out, "--fps", "5", "--cell-px", "8"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert os.path.getsize(out) > 0
