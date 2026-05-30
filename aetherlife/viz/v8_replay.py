"""Helpers purs (sans pygame) pour le V8 Replay Viewer.

Contrat events.jsonl + couleurs déterministes token/lignée.
"""
from __future__ import annotations

import colorsys
import json
from typing import Any, Iterator

SCHEMA_VERSION = 1

# 4 couleurs saturées distinctes (rouge, bleu, vert, jaune) — 1 par token.
TOKEN_COLORS: list[tuple[int, int, int]] = [
    (231, 76, 60),
    (46, 134, 222),
    (46, 204, 113),
    (241, 196, 15),
]


def token_color(token_id: int) -> tuple[int, int, int]:
    """Couleur d'un token (wrap modulo nb de couleurs)."""
    return TOKEN_COLORS[int(token_id) % len(TOKEN_COLORS)]


def lineage_color(root_id: int) -> tuple[int, int, int]:
    """Couleur stable et déterministe d'une lignée (hash root_id → teinte).

    Espacement par nombre d'or pour maximiser la séparation des teintes.
    """
    hue = (int(root_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.92)
    return (int(r * 255), int(g * 255), int(b * 255))
