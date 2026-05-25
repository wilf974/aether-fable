"""V8-B2.2 — LanguageCausalityTracker.

Observer pur (jamais d'influence sur les agents). Enregistre :
    - chaque émission de token (tick, speaker, token_id, listeners, contexte)
    - actions de chaque agent (pour baseline)

À la fin du run, calcule deux métriques causales :

M1 — listener_behavior_shift_after_token
    Pour chaque token X, KL(distribution actions post-écoute || baseline)

M2 — same_token_same_context_rate
    Pour chaque token X, % d'émissions dans un cluster contextuel
    majoritaire (food_visible, low_energy, etc.)

Si M1 > 0.10 ET M2 > 0.50 → l'hypothèse "communication causale" est
renforcée. Sinon, le pattern de divergence linguistique peut être
purement décoratif.

Spec : `docs/superpowers/specs/2026-05-25-aetherlife-v8-b2-2-language-causality-design.md`
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class _Emission:
    tick: int
    speaker_id: int
    token_id: int
    listener_ids: list[int]
    context: dict[str, Any]


@dataclass
class LanguageCausalityTracker:
    """Observer pur des événements de communication."""

    n_tokens: int
    n_actions: int
    post_listen_window: int = 3
    # State (rempli en cours de run)
    emissions: list[_Emission] = field(default_factory=list)
    # actions_by_tick[tick][agent_id] = action_id
    actions_by_tick: dict[int, dict[int, int]] = field(default_factory=dict)

    def push_emission(
        self, tick: int, speaker_id: int, token_id: int,
        listener_ids: list[int], context: dict[str, Any],
    ) -> None:
        """Enregistre une émission de token + ses listeners + contexte."""
        if 0 <= token_id < self.n_tokens:
            self.emissions.append(_Emission(
                tick=tick, speaker_id=speaker_id, token_id=token_id,
                listener_ids=list(listener_ids), context=dict(context),
            ))

    def push_actions(self, tick: int, agent_actions: dict[int, int]) -> None:
        """Enregistre les actions de tous les agents à un tick donné."""
        self.actions_by_tick[tick] = dict(agent_actions)

    # ─── Métrique M1 : listener behavior shift ─────────────────────────

    def _baseline_action_distribution(self) -> np.ndarray:
        """Distribution baseline des actions (tous agents, tous ticks)."""
        counts = np.zeros(self.n_actions, dtype=np.float64)
        for actions in self.actions_by_tick.values():
            for a in actions.values():
                if 0 <= a < self.n_actions:
                    counts[a] += 1
        total = counts.sum()
        if total == 0:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return counts / total

    def _post_listen_counts(self, token_id: int) -> np.ndarray:
        """Compte brut des actions post-écoute (sans normalisation)."""
        counts = np.zeros(self.n_actions, dtype=np.float64)
        for e in self.emissions:
            if e.token_id != token_id:
                continue
            for dt in range(1, self.post_listen_window + 1):
                t = e.tick + dt
                if t not in self.actions_by_tick:
                    continue
                for listener_id in e.listener_ids:
                    if listener_id in self.actions_by_tick[t]:
                        a = self.actions_by_tick[t][listener_id]
                        if 0 <= a < self.n_actions:
                            counts[a] += 1
        return counts

    def listener_shift_per_token(self) -> dict[int, float]:
        """KL(p_after_token || p_baseline) pour chaque token.

        Renvoie 0.0 si aucune donnée post-écoute (token jamais émis ou
        pas d'auditeurs valides) — un manque de données n'est PAS une
        divergence statistique.
        """
        baseline = self._baseline_action_distribution()
        out: dict[int, float] = {}
        for tok in range(self.n_tokens):
            counts = self._post_listen_counts(tok)
            total = counts.sum()
            if total == 0:
                out[tok] = 0.0
                continue
            p = counts / total
            out[tok] = float(_kl_divergence(p, baseline))
        return out

    # ─── Métrique M2 : same token same context rate ────────────────────

    def context_consistency_per_token(self) -> dict[int, float]:
        """% d'émissions du token dans le cluster contextuel majoritaire.

        Cluster = tuple des features contextuelles binarisées.
        """
        out: dict[int, float] = {}
        for tok in range(self.n_tokens):
            ctx_keys: list[tuple] = []
            for e in self.emissions:
                if e.token_id != tok:
                    continue
                # Convertir contexte en tuple binarisé
                key = tuple(sorted(
                    (k, _bin(v)) for k, v in e.context.items()
                ))
                ctx_keys.append(key)
            if not ctx_keys:
                out[tok] = 0.0
                continue
            counter = Counter(ctx_keys)
            most_common_n = counter.most_common(1)[0][1]
            out[tok] = most_common_n / len(ctx_keys)
        return out

    # ─── Finalisation ──────────────────────────────────────────────────

    def finalize(self) -> dict[str, Any]:
        """Renvoie dict prêt à embed dans le report final."""
        shift = self.listener_shift_per_token()
        consistency = self.context_consistency_per_token()
        n_listeners_total = sum(len(e.listener_ids) for e in self.emissions)
        out = {
            "listener_shift_per_token": {str(k): v for k, v in shift.items()},
            "listener_shift_mean": (
                float(np.mean(list(shift.values()))) if shift else 0.0
            ),
            "listener_shift_max": (
                float(np.max(list(shift.values()))) if shift else 0.0
            ),
            "context_consistency_per_token": {
                str(k): v for k, v in consistency.items()
            },
            "context_consistency_mean": (
                float(np.mean(list(consistency.values()))) if consistency else 0.0
            ),
            "n_emissions_total": len(self.emissions),
            "n_listeners_total": n_listeners_total,
            "n_action_ticks": len(self.actions_by_tick),
        }
        out["verdict"] = _verdict(out)
        return out


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _bin(v: Any) -> Any:
    """Binarise une valeur contextuelle pour clustering simple."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        # Discretise en 3 buckets : low/mid/high
        if v < 0.33:
            return "low"
        if v < 0.66:
            return "mid"
        return "high"
    return v


def _verdict(metrics: dict[str, Any]) -> str:
    """Verdict textuel basé sur les seuils définis dans la spec."""
    shift = metrics.get("listener_shift_mean", 0.0)
    cons = metrics.get("context_consistency_mean", 0.0)
    if shift > 0.10 and cons > 0.50:
        return "communication_causale_renforcee"
    if shift > 0.10 or cons > 0.50:
        return "ambigu_plus_de_seeds"
    if shift < 0.02 and cons < 0.30:
        return "decoratif_hypothese_tombe"
    return "intermediaire"
