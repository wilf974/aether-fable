"""RandomAgent — baseline sanity check."""
from __future__ import annotations

import numpy as np


class RandomAgent:
    """Agent aléatoire uniforme sur les actions."""

    def __init__(self, n_actions: int = 4, *, seed: int = 0) -> None:
        self._n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, *, info: dict | None = None) -> int:
        return int(self._rng.integers(0, self._n_actions))
