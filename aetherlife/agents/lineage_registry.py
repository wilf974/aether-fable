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
    """Index des cerveaux actifs par root_ancestor_id.

    V8-B1.5 — soft cull avec `grace_ticks` : un cerveau de lignée éteinte
    n'est pas immédiatement libéré, il reste disponible pendant
    `grace_ticks` ticks pour permettre la résurrection de la lignée
    (et donc la récupération du savoir).
    """

    def __init__(
        self,
        cfg: BrainConfig,
        obs_dim: int,
        n_actions: int,
        *,
        grace_ticks: int = 0,
        seed_bank_max_per_affinity: int = 2,
    ) -> None:
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.grace_ticks = grace_ticks
        self._brains: dict[int, LineageBrain] = {}
        # V8-B1.5 — tick d'extinction par lignée (None = vivante)
        self._extinction_ticks: dict[int, int] = {}
        # V8-B1.7 — seed bank par affinity (FIFO bornée)
        self.seed_bank_max_per_affinity = seed_bank_max_per_affinity
        self._seed_bank: dict[int, list[LineageBrain]] = {}

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

    def cull_dead_lineages(
        self,
        alive_roots: set[int],
        current_tick: int | None = None,
    ) -> int:
        """V8-B1.5 — Cull soft avec grace_ticks (si > 0).

        Logique :
            - Si `grace_ticks == 0` (V8-B1) : suppression immédiate
              des lignées éteintes (compat backward).
            - Si `grace_ticks > 0` (V8-B1.5) : on marque la date
              d'extinction et on supprime SEULEMENT après que le délai
              soit écoulé. Une lignée ressuscitée (alive_roots la contient
              de nouveau) voit son extinction_tick effacé.

        Args:
            alive_roots: set des roots vivants à ce tick.
            current_tick: tick courant. Requis si grace_ticks > 0.

        Returns:
            nombre de brains réellement supprimés.
        """
        # Reset extinction pour lignées ressuscitées
        for r in list(self._extinction_ticks):
            if r in alive_roots:
                del self._extinction_ticks[r]
        # Compat backward : cull immédiat si grace_ticks=0
        if self.grace_ticks <= 0:
            dead = [r for r in self._brains if r not in alive_roots]
            for r in dead:
                brain = self._brains.pop(r)
                self._archive_to_seed_bank(brain)  # V8-B1.7
                self._extinction_ticks.pop(r, None)
            return len(dead)
        # Soft cull : marquer extinctions nouvelles
        if current_tick is None:
            raise ValueError(
                "current_tick requis quand grace_ticks > 0"
            )
        for r in list(self._brains):
            if r not in alive_roots and r not in self._extinction_ticks:
                self._extinction_ticks[r] = current_tick
        # Free seulement après grace_ticks écoulés
        to_free = [
            r for r, t in self._extinction_ticks.items()
            if (current_tick - t) >= self.grace_ticks
        ]
        for r in to_free:
            brain = self._brains.pop(r, None)
            if brain is not None:
                self._archive_to_seed_bank(brain)  # V8-B1.7
            del self._extinction_ticks[r]
        return len(to_free)

    # V8-B1.7 — Seed bank management
    def _archive_to_seed_bank(self, brain: LineageBrain) -> None:
        """Archive un brain de lignée éteinte par affinity.

        Si l'affinity du brain est None, ne archive pas (compat V8-B1).
        FIFO bornée à `seed_bank_max_per_affinity` par affinity (le plus
        récent remplace le plus ancien).
        """
        aff = brain.biome_affinity
        if aff is None:
            return
        bank = self._seed_bank.setdefault(aff, [])
        bank.append(brain)
        while len(bank) > self.seed_bank_max_per_affinity:
            bank.pop(0)

    def get_seed_brain_for_affinity(
        self, affinity: int,
    ) -> LineageBrain | None:
        """Retourne le brain le plus récent archivé pour cette affinity.

        Ne retire pas le brain de la seed bank (réutilisable).
        Retourne None si la seed bank est vide pour cette affinity.
        """
        bank = self._seed_bank.get(affinity)
        if not bank:
            return None
        return bank[-1]

    def seed_bank_size(self, affinity: int | None = None) -> int:
        """Nombre de brains archivés (pour une affinity ou total)."""
        if affinity is None:
            return sum(len(v) for v in self._seed_bank.values())
        return len(self._seed_bank.get(affinity, []))

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
