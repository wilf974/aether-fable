"""GreedyAgent — heuristique oracle vers la food la plus proche.

Décode l'observation pour retrouver la grille food et la position agent,
puis renvoie l'action qui réduit la distance Manhattan à la food la plus proche.
Sert de borne supérieure heuristique pour V1.
"""
from __future__ import annotations

import numpy as np


class GreedyAgent:
    """Agent oracle qui se dirige vers la food la plus proche (Manhattan)."""

    def __init__(self, rows: int, cols: int, *, seed: int = 0) -> None:
        self._rows = rows
        self._cols = cols
        self._n_cells = rows * cols
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, *, info: dict | None = None) -> int:
        pos_one_hot = observation[: self._n_cells]
        food_flat = observation[self._n_cells : 2 * self._n_cells]
        pos_idx = int(np.argmax(pos_one_hot))
        agent_r, agent_c = divmod(pos_idx, self._cols)

        food_indices = np.flatnonzero(food_flat > 0.5)
        if len(food_indices) == 0:
            return int(self._rng.integers(0, 4))

        rows_arr = food_indices // self._cols
        cols_arr = food_indices % self._cols
        distances = np.abs(rows_arr - agent_r) + np.abs(cols_arr - agent_c)
        nearest = int(np.argmin(distances))
        target_r, target_c = int(rows_arr[nearest]), int(cols_arr[nearest])

        dr = target_r - agent_r
        dc = target_c - agent_c

        if abs(dr) >= abs(dc) and dr != 0:
            return 0 if dr < 0 else 1  # NORTH si dr<0, SOUTH si dr>0
        if dc != 0:
            return 2 if dc < 0 else 3  # WEST si dc<0, EAST si dc>0
        return int(self._rng.integers(0, 4))
