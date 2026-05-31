"""Mobilité spatiale — helper pur pour l'Historien (chantier A).

Mesure si la zone d'agrégation des agents reste fixe (« bassin village ») ou
se relocalise au cours d'un run, via la corrélation d'occupation entre la
fenêtre de début et la fenêtre de fin.

`mobility_score = corr_occupation_start_end` (continu, ∈ [-1, 1] ; ~1 = village
sédentaire stable, plus bas = mobilité collective). Pas de classes discrètes —
score continu + flag bassin (cf. finding 2026-05-30 coordination-mobility-modes).

Pur : aucune dépendance torch/env. overnight_v8b1 accumule, le détecteur lit.
"""
from __future__ import annotations

import statistics as st
from typing import Any

BINS = 8  # grille BINS×BINS de super-cellules
VILLAGE_BASIN_THRESHOLD = 0.8


def pearson_corr(x: list[float], y: list[float]) -> float:
    """Corrélation de Pearson. Renvoie 0.0 si variance nulle (indéfinie).

    Invariante à l'échelle → marche directement sur des comptes bruts.
    """
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def is_village_basin(corr: float) -> bool:
    """True si la corrélation d'occupation place le run dans le bassin village."""
    return corr >= VILLAGE_BASIN_THRESHOLD


class OccupancyAccumulator:
    """Accumule l'occupation spatiale en comptes par super-cellule (O(1) mémoire).

    Online : on ajoute les positions vivantes tick par tick. Pas de stockage
    de trajectoire — juste un histogramme BINS×BINS.
    """

    def __init__(self, rows: int, cols: int, bins: int = BINS) -> None:
        self.rows = rows
        self.cols = cols
        self.bins = bins
        self.counts = [0] * (bins * bins)
        self.n = 0

    def _bin(self, r: int, c: int) -> int:
        br = min(self.bins - 1, int(r * self.bins / self.rows))
        bc = min(self.bins - 1, int(c * self.bins / self.cols))
        return br * self.bins + bc

    def add_positions(self, positions: list[tuple[int, int]]) -> None:
        for r, c in positions:
            self.counts[self._bin(r, c)] += 1
            self.n += 1


def build_spatial_mobility_block(
    start: OccupancyAccumulator,
    end: OccupancyAccumulator,
    *,
    start_window: tuple[int, int],
    end_window: tuple[int, int],
) -> dict[str, Any]:
    """Construit le bloc `spatial_mobility_v8c3` du report.

    corr=None si l'une des fenêtres est vide (ex. extinction avant la fin).
    """
    if start.n == 0 or end.n == 0:
        corr: float | None = None
        village: bool | None = None
    else:
        corr = round(pearson_corr(start.counts, end.counts), 3)
        village = is_village_basin(corr)
    return {
        "corr_occupation_start_end": corr,
        "village_basin": village,
        "start_window_ticks": list(start_window),
        "end_window_ticks": list(end_window),
        "n_samples_start": start.n,
        "n_samples_end": end.n,
    }
