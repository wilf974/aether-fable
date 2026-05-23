"""Render helper pour afficher un EpisodeReport en overlay pygame.

Utilisation typique dans une GUI :

    from aetherlife.viz.report_overlay import render_report_overlay

    if showing_report:
        render_report_overlay(screen, report_lines, font_lg, font_sm)
"""
from __future__ import annotations

import pygame

OVERLAY_BG = (12, 12, 16)
OVERLAY_BORDER = (90, 200, 90)
OVERLAY_FG = (220, 220, 230)
OVERLAY_DIM = (140, 140, 150)
OVERLAY_HEADER = (240, 200, 100)
OVERLAY_TIP = (90, 200, 90)


def render_report_overlay(
    screen: pygame.Surface,
    lines: list[str],
    font_lg: pygame.font.Font,
    font_sm: pygame.font.Font,
    *,
    tip: str = "SPACE / R / N to continue",
    padding: int = 16,
    line_h: int = 18,
    alpha: int = 230,
) -> None:
    """Dessine un panneau semi-transparent centré contenant les lignes du rapport."""
    w, h = screen.get_size()
    content_w = min(w - 60, max(420, max(len(s) for s in lines) * 8))
    content_h = padding * 2 + line_h * (len(lines) + 2)
    content_h = min(content_h, h - 60)

    x = (w - content_w) // 2
    y = (h - content_h) // 2

    panel = pygame.Surface((content_w, content_h), pygame.SRCALPHA)
    panel.fill((*OVERLAY_BG, alpha))
    pygame.draw.rect(panel, OVERLAY_BORDER, panel.get_rect(), 2)

    cur_y = padding
    for i, line in enumerate(lines):
        if i == 0:
            surf = font_lg.render(line, True, OVERLAY_HEADER)
        elif line.lstrip().startswith("--"):
            surf = font_sm.render(line, True, OVERLAY_DIM)
        elif line.lstrip().startswith("#"):
            surf = font_sm.render(line, True, OVERLAY_FG)
        else:
            surf = font_sm.render(line, True, OVERLAY_FG)
        panel.blit(surf, (padding, cur_y))
        cur_y += line_h
        if cur_y > content_h - padding - line_h:
            break

    # tip en bas
    tip_surf = font_sm.render(tip, True, OVERLAY_TIP)
    panel.blit(tip_surf, (padding, content_h - padding - line_h))

    screen.blit(panel, (x, y))
