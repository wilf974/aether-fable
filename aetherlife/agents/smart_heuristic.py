"""SmartHeuristicAgent — politique simple mais stratégique pour env V5+ saisonnier.

Comportement déterministe basé sur l'état de l'agent + carte de l'env. Pas du
RL, juste des règles claires pour faire émerger des comportements visibles
dans la GUI (recherche food, retour au nid, construction, dépôt cache,
retrait cache).

Hiérarchie d'objectifs (par priorité décroissante) :
    1. Famine (energy < critical) ET il a un nid avec cache > 0 → va au nid
    2. Famine (energy < critical) → va à la food la plus proche (Manhattan)
    3. Surplus (energy ≥ build_threshold) ET PAS de nid → reste sur place
       (si cellule libre de food) pour construire au prochain step
    4. Surplus ET nid existe → va au nid (pour rest + déposer dans cache)
    5. Énergie moyenne ET food visible proche → va manger
    6. Sinon : marche aléatoire douce (préfère ne pas revenir au step
       précédent)
"""
from __future__ import annotations

import numpy as np

from aetherlife.world.food_grid import Action


class SmartHeuristicAgent:
    """Agent heuristique cheat-friendly (lit env complet).

    Args:
        env: env saisonnier (a `food_mask`, `agent_state(id)`, `nests`)
        seed: RNG pour les tirages aléatoires (tiebreaks, walk).
        critical_energy: seuil sous lequel l'agent va chercher de la food
            ou de son cache de toute urgence.
        comfortable_energy: seuil au-dessus duquel l'agent priorise la
            construction / le rest plutôt que la chasse.
    """

    def __init__(
        self,
        env,
        *,
        seed: int = 0,
        critical_energy_factor: float = 0.35,
        comfortable_energy_factor: float = 0.7,
    ) -> None:
        self.env = env
        self._rng = np.random.default_rng(seed)
        self.critical_energy_factor = critical_energy_factor
        self.comfortable_energy_factor = comfortable_energy_factor
        self._prev_pos: dict[int, tuple[int, int]] = {}

    def act_dict(
        self,
        obs_dict: dict[int, np.ndarray],
        *,
        greedy: bool = True,
    ) -> dict[int, int]:
        actions: dict[int, int] = {}
        env = self.env
        critical = env.cfg.max_energy * self.critical_energy_factor
        comfortable = env.cfg.max_energy * self.comfortable_energy_factor
        build_threshold = (
            env.cfg.build.energy_threshold
            if hasattr(env.cfg, "build") and env.cfg.build.enabled
            else float("inf")
        )
        food_positions = self._food_positions()
        nest_lookup = env.nests if hasattr(env, "nests") else {}
        nest_pos_set = {n.pos for n in nest_lookup.values()} if nest_lookup else set()

        for aid in obs_dict:
            try:
                agent = env.agent_state(aid)
            except Exception:
                actions[aid] = int(self._rng.integers(0, 4))
                continue
            if not agent.alive:
                actions[aid] = int(self._rng.integers(0, 4))
                continue

            ar, ac = agent.pos
            energy = agent.energy

            # 1. Famine critique : aller à son nid (si cache) ou food
            own_nest = nest_lookup.get(aid)
            if energy < critical:
                if own_nest is not None:
                    # Si le nid a du cache, prioriser le nid
                    cache_stock = getattr(env, "nest_food_stock", {})
                    if cache_stock.get(aid, 0) > 0:
                        actions[aid] = self._move_toward(ar, ac, *own_nest.pos)
                        self._prev_pos[aid] = (ar, ac)
                        continue
                # Sinon, chasser la food la plus proche
                target = self._nearest_food(ar, ac, food_positions)
                if target is not None:
                    actions[aid] = self._move_toward(ar, ac, *target)
                    self._prev_pos[aid] = (ar, ac)
                    continue
                # Pas de food visible, marche aléatoire douce
                actions[aid] = self._random_walk(aid, ar, ac)
                continue

            # 2. Surplus + pas de nid → construire (rester sur place libre)
            if energy >= build_threshold and own_nest is None:
                if env.food_mask[ar, ac]:
                    actions[aid] = self._random_walk(aid, ar, ac)
                else:
                    actions[aid] = self._idle_action(ar, ac)
                self._prev_pos[aid] = (ar, ac)
                continue

            # V6.1 — Plantation intentionnelle avec système de graines.
            # Logique de raisonnement :
            #   - Pas de graine → impossible de planter, doit chasser food
            #   - Graine + énergie suffisante + nid → plantation prioritaire
            #     surtout si silo bas / hiver approche
            pcfg = getattr(env.cfg, "planting", None)
            plants = getattr(env, "plants", {})
            has_seed = agent.seeds >= (
                pcfg.seeds_required if pcfg and pcfg.enabled else 1
            )
            can_plant = (
                pcfg is not None and pcfg.enabled
                and energy >= pcfg.energy_threshold
                and has_seed
            )
            # Détection "silo bas" et "hiver approche" pour augmenter prio plant
            cache_low = False
            if pcfg and pcfg.enabled and own_nest is not None:
                cache_stock = getattr(env, "nest_food_stock", {})
                stock = cache_stock.get(own_nest.owner_id, 0)
                cap = (
                    env.cfg.cache.max_capacity
                    if hasattr(env.cfg, "cache") and env.cfg.cache.enabled
                    else 1.0
                )
                cache_low = (cap > 0) and (stock / cap < 0.4)
            winter_imminent = False
            if hasattr(env, "phase"):
                phase = env.phase
                # winter = phase ∈ [0.75, 1.0)
                # imminent si phase ∈ [0.6, 0.75)
                winter_imminent = 0.6 <= phase < 0.75
            should_prioritize_plant = (
                can_plant and (cache_low or winter_imminent or own_nest is not None)
            )
            if should_prioritize_plant:
                if (
                    not env.food_mask[ar, ac]
                    and (ar, ac) not in plants
                    and (ar, ac) not in nest_pos_set
                ):
                    actions[aid] = self._idle_action(ar, ac)
                    self._prev_pos[aid] = (ar, ac)
                    continue
                tgt = self._nearest_plantable(ar, ac, env, plants, nest_pos_set)
                if tgt is not None:
                    actions[aid] = self._move_toward(ar, ac, *tgt)
                    self._prev_pos[aid] = (ar, ac)
                    continue

            # 3. Surplus ET nid existe → aller au nid (rest + déposer cache)
            if energy >= comfortable and own_nest is not None:
                if (ar, ac) == own_nest.pos:
                    actions[aid] = self._idle_action(ar, ac)
                else:
                    actions[aid] = self._move_toward(ar, ac, *own_nest.pos)
                self._prev_pos[aid] = (ar, ac)
                continue

            # 4. Énergie moyenne : chercher food si proche
            target = self._nearest_food(ar, ac, food_positions, max_dist=10)
            if target is not None:
                actions[aid] = self._move_toward(ar, ac, *target)
                self._prev_pos[aid] = (ar, ac)
                continue

            # 5. Sinon : marche aléatoire (préfère ne pas revenir)
            actions[aid] = self._random_walk(aid, ar, ac)
            self._prev_pos[aid] = (ar, ac)

        return actions

    # ─── helpers ─────────────────────────────────────────────────────────
    def _food_positions(self) -> list[tuple[int, int]]:
        mask = self.env.food_mask
        rs, cs = np.where(mask)
        return [(int(rs[i]), int(cs[i])) for i in range(len(rs))]

    def _nearest_food(
        self, ar: int, ac: int, foods: list[tuple[int, int]],
        max_dist: int | None = None,
    ) -> tuple[int, int] | None:
        if not foods:
            return None
        best = None
        best_d = float("inf")
        for fr, fc in foods:
            d = abs(fr - ar) + abs(fc - ac)
            if max_dist is not None and d > max_dist:
                continue
            if d < best_d:
                best_d = d
                best = (fr, fc)
        return best

    def _move_toward(self, ar: int, ac: int, tr: int, tc: int) -> int:
        dr = tr - ar
        dc = tc - ac
        # Priorité à l'axe le plus long
        if abs(dr) >= abs(dc) and dr != 0:
            return int(Action.NORTH) if dr < 0 else int(Action.SOUTH)
        if dc != 0:
            return int(Action.WEST) if dc < 0 else int(Action.EAST)
        return self._idle_action(ar, ac)

    def _nearest_plantable(
        self, ar: int, ac: int, env, plants: dict,
        nest_pos_set: set[tuple[int, int]],
        max_dist: int = 6,
    ) -> tuple[int, int] | None:
        """V6.1 — trouve la cellule plantable la plus proche (libre de food/nid/plant)."""
        rows, cols = env.cfg.rows, env.cfg.cols
        best = None
        best_d = float("inf")
        for r in range(max(0, ar - max_dist), min(rows, ar + max_dist + 1)):
            for c in range(max(0, ac - max_dist), min(cols, ac + max_dist + 1)):
                if env.food_mask[r, c]:
                    continue
                if (r, c) in plants:
                    continue
                if (r, c) in nest_pos_set:
                    continue
                d = abs(r - ar) + abs(c - ac)
                if d == 0:
                    continue
                if d < best_d:
                    best_d = d
                    best = (r, c)
        return best

    def _idle_action(self, ar: int, ac: int) -> int:
        """Action qui clamp et fait rester l'agent sur place (selon position)."""
        if ar == 0:
            return int(Action.NORTH)
        if ar == self.env.cfg.rows - 1:
            return int(Action.SOUTH)
        if ac == 0:
            return int(Action.WEST)
        if ac == self.env.cfg.cols - 1:
            return int(Action.EAST)
        # pas au bord : pas d'idle pur disponible, choisir aléatoire
        return int(self._rng.integers(0, 4))

    def _random_walk(self, aid: int, ar: int, ac: int) -> int:
        prev = self._prev_pos.get(aid)
        # Évite de revenir directement à la cellule précédente
        candidates = [0, 1, 2, 3]
        if prev is not None:
            pr, pc = prev
            dr, dc = pr - ar, pc - ac
            if dr == -1 and dc == 0:
                candidates = [a for a in candidates if a != int(Action.NORTH)]
            elif dr == 1 and dc == 0:
                candidates = [a for a in candidates if a != int(Action.SOUTH)]
            elif dr == 0 and dc == -1:
                candidates = [a for a in candidates if a != int(Action.WEST)]
            elif dr == 0 and dc == 1:
                candidates = [a for a in candidates if a != int(Action.EAST)]
        return int(self._rng.choice(candidates))
