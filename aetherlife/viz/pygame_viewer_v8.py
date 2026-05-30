"""Renderer V8 Replay Viewer — pur (lit le contrat, pas de torch/env).

SDL dummy par défaut : rend en offscreen, aucune fenêtre requise.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from aetherlife.viz.v8_replay import (  # noqa: E402
    iter_events, lineage_color, load_meta, token_color,
)

BG = (18, 18, 20)
SPOT = (120, 120, 130)
SPOT_HOT = (255, 230, 120)
HUD_H = 28
HUD_BG = (10, 10, 12)
HUD_FG = (210, 210, 215)


def _draw_frame(
    event: dict, meta: dict, cell_px: int = 16, focus_lineage: int | None = None
) -> "pygame.Surface":
    """Rend un tick en une Surface offscreen."""
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    rows, cols = int(meta["rows"]), int(meta["cols"])
    width, height = cols * cell_px, rows * cell_px + HUD_H
    surf = pygame.Surface((width, height))
    surf.fill(BG)

    # Gather spots (sous les agents) — surbrillance si >= 2 adjacents
    for s in event.get("spots", []):
        hot = int(s.get("n", 0)) >= 2
        rect = pygame.Rect(s["c"] * cell_px, s["r"] * cell_px, cell_px, cell_px)
        pygame.draw.rect(surf, SPOT_HOT if hot else SPOT, rect, width=2)

    # Agents teintés par lignée
    pos_by_id: dict[int, tuple[int, int]] = {}
    for a in event["agents"]:
        col = lineage_color(a["lin"])
        if focus_lineage is not None and a["lin"] != focus_lineage:
            col = (col[0] // 3, col[1] // 3, col[2] // 3)
        rect = pygame.Rect(a["c"] * cell_px, a["r"] * cell_px, cell_px, cell_px)
        pygame.draw.rect(surf, col, rect)
        pos_by_id[int(a["id"])] = (a["r"], a["c"])

    # Halos de vocalisation (couleur = token), au sommet de la cellule du speaker
    radius = max(2, cell_px // 4)
    for sid, tok in event.get("vocal", {}).items():
        rc = pos_by_id.get(int(sid))
        if rc is None:
            continue
        cx = rc[1] * cell_px + cell_px // 2
        cy = rc[0] * cell_px + radius
        pygame.draw.circle(surf, token_color(tok), (cx, cy), radius)

    # HUD
    pygame.draw.rect(surf, HUD_BG, pygame.Rect(0, rows * cell_px, width, HUD_H))
    font = pygame.font.SysFont("monospace", 14)
    txt = (
        f"t={event.get('t', 0)}  alive={event.get('n_alive', '?')}  "
        f"lin={event.get('n_lin', '?')}  season={event.get('season', '?')}"
    )
    surf.blit(font.render(txt, True, HUD_FG), (6, rows * cell_px + 6))
    return surf
