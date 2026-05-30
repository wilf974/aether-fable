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


def load_meta(path: str) -> dict[str, Any]:
    """Charge meta.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_events(path: str) -> Iterator[dict[str, Any]]:
    """Itère les événements d'un events.jsonl (1 objet JSON par ligne non vide)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_REQUIRED_EVENT_KEYS = {"t", "agents"}


def validate_event(event: dict[str, Any]) -> bool:
    """Vérifie la présence des clés minimales du contrat."""
    missing = _REQUIRED_EVENT_KEYS - set(event.keys())
    if missing:
        raise ValueError(f"event missing keys: {sorted(missing)}")
    return True
