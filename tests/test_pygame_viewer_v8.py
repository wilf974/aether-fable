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
