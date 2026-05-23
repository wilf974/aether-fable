"""V8-B1.1 — LineageRegistry : index global des cerveaux par root_ancestor_id.

Un registry centralise tous les `LineageBrain` actifs, indexés par leur
`root_ancestor_id`. Il fournit :

    - `get_or_create(root_id, parent_brain)` : crée un brain s'il n'existe
      pas (via héritage si parent_brain fourni, ou init random sinon).
    - `cull_dead_lineages(alive_roots)` : libère les brains des lignées
      éteintes (libère la VRAM GPU si applicable).
    - `act_for_lineage(root_id, obs)` : raccourci pour brain.act() via
      lookup.

Vie d'un brain :
    1. Premier agent d'une lignée → registry crée le brain
    2. Tous les descendants de cette lignée partagent le même brain
       (1 cerveau par root_ancestor_id, pas par agent)
    3. Quand le dernier agent de la lignée meurt → cull retire le brain

Note V8-B1 : on n'a pas de fork intra-lignée pour le moment. Si on veut
diverger un sous-arbre, on devra implémenter `fork_subtree(parent_brain, new_root_id)`
en B2/B3.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from aetherlife.agents.lineage_brain import BrainConfig, LineageBrain


class LineageRegistry:
    """Index des cerveaux actifs par root_ancestor_id."""

    def __init__(
        self,
        cfg: BrainConfig,
        obs_dim: int,
        n_actions: int,
    ) -> None:
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self._brains: dict[int, LineageBrain] = {}

    def __len__(self) -> int:
        return len(self._brains)

    def __contains__(self, root_id: int) -> bool:
        return root_id in self._brains

    def __iter__(self) -> Iterator[LineageBrain]:
        return iter(self._brains.values())

    def get(self, root_id: int) -> LineageBrain | None:
        """Retourne le brain de la lignée ou None si absent."""
        return self._brains.get(root_id)

    def get_or_create(
        self,
        root_id: int,
        parent_brain: LineageBrain | None = None,
        *,
        seed: int = 0,
    ) -> LineageBrain:
        """Retourne le brain de la lignée, créant-le si nécessaire.

        Si `parent_brain` fourni, le nouveau brain hérite de ses poids
        avec mutation gaussienne (std = cfg.mutation_std).
        Sinon, init random.
        """
        existing = self._brains.get(root_id)
        if existing is not None:
            return existing
        if parent_brain is not None:
            brain = LineageBrain.inherit_from(
                parent=parent_brain, root_id=root_id,
                mutation_std=self.cfg.mutation_std, seed=seed,
            )
        else:
            brain = LineageBrain(
                root_id=root_id,
                obs_dim=self.obs_dim,
                n_actions=self.n_actions,
                cfg=self.cfg,
                seed=seed,
            )
        self._brains[root_id] = brain
        return brain

    def cull_dead_lineages(self, alive_roots: set[int]) -> int:
        """Supprime les brains des lignées éteintes. Retourne n supprimés."""
        dead = [r for r in self._brains if r not in alive_roots]
        for r in dead:
            del self._brains[r]
        return len(dead)

    def alive_roots(self) -> set[int]:
        return set(self._brains.keys())

    def total_global_steps(self) -> int:
        return sum(b.global_step for b in self._brains.values())

    def act_for_lineage(
        self,
        root_id: int,
        obs: np.ndarray,
        *,
        greedy: bool = False,
    ) -> int:
        """Raccourci : lookup brain + act. Lève si lignée inconnue."""
        brain = self._brains[root_id]
        return brain.act(obs, greedy=greedy)
