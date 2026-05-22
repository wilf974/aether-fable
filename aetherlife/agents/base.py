"""Protocole de base pour les agents AetherLife V1."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    """Protocole minimal pour V1 — un agent doit pouvoir choisir une action."""

    def act(self, observation: np.ndarray, *, info: dict | None = None) -> int:
        """Retourne une action dans [0, n_actions)."""
        ...
