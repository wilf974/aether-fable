"""FoodGrid — environnement V1 single-agent forager.

API inspirée de Gymnasium (reset / step renvoient des tuples), mais pas de
dépendance Gym ici. Le wrapper Gymnasium est dans aetherlife.env.single_agent_env.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from aetherlife.config import FoodGridConfig
from aetherlife.guardrails.invariants import (
    clamp_pos,
    energy_no_food,
    energy_with_food,
    is_terminated,
    step_reward,
)


class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3


_DELTAS: dict[Action, tuple[int, int]] = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
    Action.EAST: (0, 1),
}


class FoodGrid:
    """Grille 2D avec un agent unique, food cells, énergie, respawn.

    State :
        - `food_mask` : np.ndarray (rows, cols) bool — True si food présent.
        - `pos` : (row, col) int — position agent.
        - `energy` : float — énergie agent.
        - `step_count` : int — ticks depuis reset.
    """

    def __init__(self, cfg: FoodGridConfig | None = None) -> None:
        self.cfg = cfg or FoodGridConfig()
        self._food_mask: np.ndarray = np.zeros(
            (self.cfg.rows, self.cfg.cols), dtype=bool
        )
        self._pos: tuple[int, int] = self.cfg.start_position
        self._energy: float = self.cfg.start_energy
        self._step_count: int = 0
        self._env_rng: np.random.Generator = np.random.default_rng()
        self._spawn_rng: np.random.Generator = np.random.default_rng()

    @property
    def n_actions(self) -> int:
        return len(Action)

    @property
    def pos(self) -> tuple[int, int]:
        return self._pos

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def food_count(self) -> int:
        return int(self._food_mask.sum())

    @property
    def food_mask(self) -> np.ndarray:
        return self._food_mask.copy()

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Réinitialise l'env. Retourne (observation, info).

        Si `seed` est fourni, les deux RNG (env initial layout + spawn dynamique)
        sont seedés séparément à partir de la même graine pour reproductibilité.
        """
        if seed is not None:
            self._env_rng = np.random.default_rng(seed)
            self._spawn_rng = np.random.default_rng(seed + 1)
        self._pos = self.cfg.start_position
        self._energy = self.cfg.start_energy
        self._step_count = 0
        self._food_mask = self._initial_food_layout()
        return self._observation(), {"step": 0, "food_count": self.food_count}

    def step(
        self, action: Action | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step l'env. Retourne (obs, reward, terminated, truncated, info)."""
        action = Action(int(action))
        dr, dc = _DELTAS[action]
        r, c = self._pos
        new_r = clamp_pos(r, dr, self.cfg.rows)
        new_c = clamp_pos(c, dc, self.cfg.cols)
        self._pos = (new_r, new_c)

        ate = bool(self._food_mask[new_r, new_c])
        if ate:
            self._food_mask[new_r, new_c] = False
            self._energy = energy_with_food(
                self._energy,
                self.cfg.metabolism,
                self.cfg.food_value,
                self.cfg.max_energy,
            )
        else:
            self._energy = energy_no_food(self._energy, self.cfg.metabolism)

        self._step_count += 1
        reward = step_reward(self.cfg.metabolism, self.cfg.food_value, ate)
        terminated = is_terminated(self._energy)
        if terminated:
            reward -= self.cfg.death_penalty
        truncated = (not terminated) and self._step_count >= self.cfg.max_steps

        if not terminated:
            self._respawn_food()

        info: dict[str, Any] = {
            "step": self._step_count,
            "ate": ate,
            "food_count": self.food_count,
            "energy": self._energy,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _initial_food_layout(self) -> np.ndarray:
        """Place `initial_food_density * rows * cols` food cells aléatoirement.

        Ne place pas de food sous la position de départ de l'agent.
        """
        n_cells = self.cfg.rows * self.cfg.cols
        target = int(round(self.cfg.initial_food_density * n_cells))
        mask = np.zeros((self.cfg.rows, self.cfg.cols), dtype=bool)
        if target == 0:
            return mask
        indices = np.arange(n_cells)
        start_idx = self._pos_to_index(self.cfg.start_position)
        valid = indices[indices != start_idx]
        chosen = self._env_rng.choice(valid, size=min(target, len(valid)), replace=False)
        for idx in chosen:
            r, c = self._index_to_pos(int(idx))
            mask[r, c] = True
        return mask

    def _respawn_food(self) -> None:
        """Spawn ~Poisson(lambda) new food cells par tick sur cellules libres."""
        if self.cfg.food_respawn_lambda <= 0:
            return
        n_spawn = int(self._spawn_rng.poisson(self.cfg.food_respawn_lambda))
        if n_spawn == 0:
            return
        free_cells = self._free_cells()
        if len(free_cells) == 0:
            return
        n_actual = min(n_spawn, len(free_cells))
        choice = self._spawn_rng.choice(len(free_cells), size=n_actual, replace=False)
        for c_idx in choice:
            r, c = free_cells[int(c_idx)]
            self._food_mask[r, c] = True

    def _free_cells(self) -> list[tuple[int, int]]:
        """Cellules sans food et qui ne contiennent pas l'agent."""
        free: list[tuple[int, int]] = []
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                if not self._food_mask[r, c] and (r, c) != self._pos:
                    free.append((r, c))
        return free

    def _observation(self) -> np.ndarray:
        """Observation = concat(pos_one_hot, food_flat, energy_normalized)."""
        n_cells = self.cfg.rows * self.cfg.cols
        obs = np.zeros(2 * n_cells + 1, dtype=np.float32)
        pos_idx = self._pos_to_index(self._pos)
        obs[pos_idx] = 1.0
        obs[n_cells : 2 * n_cells] = self._food_mask.flatten().astype(np.float32)
        obs[-1] = float(self._energy) / float(self.cfg.max_energy)
        return obs

    def _pos_to_index(self, p: tuple[int, int]) -> int:
        r, c = p
        return r * self.cfg.cols + c

    def _index_to_pos(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.cfg.cols)

    def render_ascii(self) -> str:
        """Rendu texte pour debug."""
        lines: list[str] = []
        for r in range(self.cfg.rows):
            row_chars: list[str] = []
            for c in range(self.cfg.cols):
                if (r, c) == self._pos:
                    row_chars.append("A")
                elif self._food_mask[r, c]:
                    row_chars.append("*")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        lines.append(f"energy={self._energy:.1f} step={self._step_count} food={self.food_count}")
        return "\n".join(lines)
