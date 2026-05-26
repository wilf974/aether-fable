"""V8-C3 — Métriques d'émergence coopérative.

4 métriques prioritaires pour détecter la proto-coordination AVANT toute
ablation de langage :

1. clustering_pre_success : densité spatiale d'agents AVANT chaque succès
   gather (les agents convergent-ils intentionnellement ?).
2. vocalize_to_gather_delay : temps moyen entre la dernière vocalize d'un
   participant et le succès collectif (un proto-protocole baisse ce délai
   au fil du temps).
3. token_entropy_pre_success : distribution des tokens vocalize dans une
   fenêtre temporelle juste AVANT chaque succès. Si un token devient
   dominant, c'est un candidat "signal coopératif".
4. success_chains : longueurs des cascades de succès consécutifs (un
   succès qui déclenche d'autres comportements coopératifs derrière).

Toutes ces mesures sont OBSERVATIONNELLES : aucune influence sur le
training, juste de la télémétrie pour décider si C3a→C3b est viable.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class GatherSuccessEvent:
    """Un succès gather_collective tel qu'observé."""
    tick: int
    pos: tuple[int, int]
    participants: list[int]  # agent_ids
    # Snapshot écosystème au moment du succès
    n_alive: int
    # Pour clustering : nombre d'agents dans rayon 3 autour du spot
    n_neighbors_r3: int
    # Pour token : tokens vocalize dans la fenêtre [tick-W, tick] par les
    # participants. Liste de (agent_id, token_id, dt_before_success).
    tokens_in_window: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class CooperativeMetricsConfig:
    """Configuration des observateurs coopératifs."""
    # Fenêtre rétrospective (en ticks) pour analyser tokens et clustering
    pre_success_window: int = 5
    # Rayon pour mesurer le clustering autour d'un spot
    cluster_radius: int = 3
    # Fenêtre pour "cascade" : succès dans X ticks suivants comptent en chaîne
    chain_window: int = 10


class CooperativeMetricsTracker:
    """Observateur passif des dynamiques coopératives V8-C3.

    Hook côté env :
      - `track_vocalize(tick, agent_id, token_id, pos)` : appel à chaque
        token émis (déjà journalisé par VocabularyConfig).
      - `track_success(tick, pos, participants_ids, alive_positions)` :
        appelé à chaque _try_gather_collective réussi.

    finalize() retourne un dict de métriques agrégées.
    """

    def __init__(self, cfg: CooperativeMetricsConfig | None = None) -> None:
        self.cfg = cfg or CooperativeMetricsConfig()
        # vocalize history bounded (on garde juste ce qu'il faut pour la
        # fenêtre — pré_success_window ticks).
        # Each entry: (tick, agent_id, token_id, (r, c))
        self._vocalize_log: deque = deque(
            maxlen=10000  # safety cap, mais purgé par tick
        )
        # All gather successes recorded
        self._successes: list[GatherSuccessEvent] = []
        # Per-agent : last vocalize tick (pour delay)
        self._last_vocalize_tick: dict[int, int] = {}
        # Pour success chains : on enregistre les ticks bruts
        self._success_ticks: list[int] = []

    def track_vocalize(
        self, tick: int, agent_id: int, token_id: int, pos: tuple[int, int],
    ) -> None:
        self._vocalize_log.append((tick, agent_id, token_id, pos))
        self._last_vocalize_tick[agent_id] = tick

    def track_success(
        self,
        tick: int,
        pos: tuple[int, int],
        participants: list[int],
        all_alive_positions: list[tuple[int, int]],
    ) -> None:
        """Enregistre un succès gather + son contexte rétrospectif."""
        W = self.cfg.pre_success_window
        R = self.cfg.cluster_radius
        # Clustering : agents (vivants) dans rayon R autour du spot
        sr, sc = pos
        n_in_r = sum(
            1 for (ar, ac) in all_alive_positions
            if abs(ar - sr) + abs(ac - sc) <= R
        )
        # Tokens dans la fenêtre [tick-W, tick] par les participants
        participants_set = set(participants)
        tokens_in_window = [
            (aid, tok, tick - t_v)
            for (t_v, aid, tok, _p) in self._vocalize_log
            if (tick - W) <= t_v <= tick and aid in participants_set
        ]
        evt = GatherSuccessEvent(
            tick=tick,
            pos=pos,
            participants=list(participants),
            n_alive=len(all_alive_positions),
            n_neighbors_r3=n_in_r,
            tokens_in_window=tokens_in_window,
        )
        self._successes.append(evt)
        self._success_ticks.append(tick)

    def prune_old_vocalize(self, current_tick: int) -> None:
        """Purge les vocalize hors fenêtre pour limiter la mémoire."""
        W = self.cfg.pre_success_window
        cutoff = current_tick - W
        while self._vocalize_log and self._vocalize_log[0][0] < cutoff:
            self._vocalize_log.popleft()

    # ---- Aggregations ----------------------------------------------------

    def _clustering_pre_success(self) -> dict:
        """Métrique 1 : clustering avant succès."""
        if not self._successes:
            return {"n": 0, "mean_neighbors_r3": 0.0}
        ns = [e.n_neighbors_r3 for e in self._successes]
        # Tendance : compare first quartile vs last quartile
        n = len(ns)
        if n >= 4:
            q1 = sum(ns[: n // 4]) / max(n // 4, 1)
            q4 = sum(ns[3 * n // 4 :]) / max(n - 3 * n // 4, 1)
            trend = q4 - q1
        else:
            trend = 0.0
        return {
            "n": len(ns),
            "mean_neighbors_r3": sum(ns) / len(ns),
            "median_neighbors_r3": sorted(ns)[len(ns) // 2],
            "trend_q4_minus_q1": float(trend),
        }

    def _vocalize_to_gather_delay(self) -> dict:
        """Métrique 2 : délai entre dernière vocalize d'un participant et le succès."""
        delays: list[int] = []
        per_success_min_delay: list[int] = []
        for e in self._successes:
            if not e.tokens_in_window:
                continue
            dts = [dt for (_a, _t, dt) in e.tokens_in_window]
            delays.extend(dts)
            per_success_min_delay.append(min(dts))
        if not per_success_min_delay:
            return {"n_with_token": 0, "mean_min_delay": None}
        # Tendance : delays sur Q1 vs Q4 (en index de succès)
        n = len(per_success_min_delay)
        if n >= 4:
            q1 = sum(per_success_min_delay[: n // 4]) / max(n // 4, 1)
            q4 = sum(per_success_min_delay[3 * n // 4 :]) / max(
                n - 3 * n // 4, 1
            )
            trend = q4 - q1  # < 0 = apprentissage (délai diminue)
        else:
            trend = 0.0
        return {
            "n_with_token": len(per_success_min_delay),
            "mean_min_delay": sum(per_success_min_delay)
            / len(per_success_min_delay),
            "median_min_delay": sorted(per_success_min_delay)[
                len(per_success_min_delay) // 2
            ],
            "trend_q4_minus_q1": float(trend),
            "coverage": len(per_success_min_delay)
            / max(len(self._successes), 1),
        }

    def _token_entropy_pre_success(self) -> dict:
        """Métrique 3 : distribution des tokens vocalize avant succès.

        Si un token devient dominant SPÉCIFIQUEMENT avant succès, c'est un
        candidat "signal coopératif".
        """
        token_counts_pre: dict[int, int] = defaultdict(int)
        n_succ_with_token = 0
        for e in self._successes:
            if not e.tokens_in_window:
                continue
            n_succ_with_token += 1
            for (_a, tok, _dt) in e.tokens_in_window:
                token_counts_pre[tok] += 1
        total = sum(token_counts_pre.values())
        if total == 0:
            return {"n": 0, "distribution": {}, "dominant_token": None}
        dist = {
            str(k): v / total for k, v in token_counts_pre.items()
        }
        dominant = max(token_counts_pre.items(), key=lambda kv: kv[1])
        # Entropy (log naturel)
        import math
        ent = 0.0
        for p in dist.values():
            if p > 0:
                ent -= p * math.log(p)
        return {
            "n_successes_with_token": n_succ_with_token,
            "n_tokens_counted": total,
            "distribution": dist,
            "dominant_token": int(dominant[0]),
            "dominant_share": dominant[1] / total,
            "entropy": ent,
        }

    def _success_chains(self) -> dict:
        """Métrique 4 : cascades de succès.

        Définition : un succès au tick t et un autre dans [t+1, t+W] sont
        considérés "enchaînés". On compte les longueurs de chaînes.
        """
        W = self.cfg.chain_window
        chains: list[int] = []
        if not self._success_ticks:
            return {"n_chains": 0, "max_chain_len": 0, "mean_chain_len": 0.0}
        ticks = sorted(self._success_ticks)
        current = 1
        for i in range(1, len(ticks)):
            if ticks[i] - ticks[i - 1] <= W:
                current += 1
            else:
                chains.append(current)
                current = 1
        chains.append(current)
        return {
            "n_chains": len(chains),
            "max_chain_len": max(chains),
            "mean_chain_len": sum(chains) / len(chains),
            "n_isolated_successes": sum(1 for c in chains if c == 1),
            "n_cascade_successes": sum(c for c in chains if c >= 3),
        }

    def finalize(self) -> dict:
        return {
            "n_successes_observed": len(self._successes),
            "clustering_pre_success": self._clustering_pre_success(),
            "vocalize_to_gather_delay": self._vocalize_to_gather_delay(),
            "token_entropy_pre_success": self._token_entropy_pre_success(),
            "success_chains": self._success_chains(),
        }
