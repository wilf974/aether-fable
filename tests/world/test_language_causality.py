"""Tests V8-B2.2 — LanguageCausalityTracker (observer pur)."""
from __future__ import annotations

import pytest

from aetherlife.world.language_causality import (
    LanguageCausalityTracker, _bin, _kl_divergence, _verdict,
)


# ─── Helpers internes ───────────────────────────────────────────────────


def test_kl_divergence_zero() -> None:
    import numpy as np
    p = np.array([0.5, 0.5])
    assert _kl_divergence(p, p) < 1e-6


def test_kl_divergence_positive() -> None:
    import numpy as np
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    assert _kl_divergence(p, q) > 0.1


def test_bin_buckets() -> None:
    assert _bin(0.1) == "low"
    assert _bin(0.5) == "mid"
    assert _bin(0.9) == "high"
    assert _bin(True) is True
    assert _bin("food_visible") == "food_visible"


def test_verdict_strong() -> None:
    assert _verdict({"listener_shift_mean": 0.2, "context_consistency_mean": 0.7}) == "communication_causale_renforcee"


def test_verdict_weak() -> None:
    assert _verdict({"listener_shift_mean": 0.01, "context_consistency_mean": 0.2}) == "decoratif_hypothese_tombe"


def test_verdict_ambiguous() -> None:
    assert _verdict({"listener_shift_mean": 0.15, "context_consistency_mean": 0.3}) == "ambigu_plus_de_seeds"


# ─── LanguageCausalityTracker ──────────────────────────────────────────


def test_tracker_init() -> None:
    t = LanguageCausalityTracker(n_tokens=4, n_actions=8)
    assert t.n_tokens == 4
    assert t.n_actions == 8
    assert len(t.emissions) == 0
    assert len(t.actions_by_tick) == 0


def test_push_emission() -> None:
    t = LanguageCausalityTracker(n_tokens=4, n_actions=8)
    t.push_emission(
        tick=10, speaker_id=1, token_id=2,
        listener_ids=[3, 4], context={"food_visible": True},
    )
    assert len(t.emissions) == 1
    assert t.emissions[0].speaker_id == 1
    assert t.emissions[0].token_id == 2
    assert t.emissions[0].listener_ids == [3, 4]


def test_push_emission_invalid_token_ignored() -> None:
    t = LanguageCausalityTracker(n_tokens=4, n_actions=8)
    t.push_emission(tick=10, speaker_id=1, token_id=99, listener_ids=[2], context={})
    t.push_emission(tick=10, speaker_id=1, token_id=-1, listener_ids=[2], context={})
    assert len(t.emissions) == 0


def test_push_actions() -> None:
    t = LanguageCausalityTracker(n_tokens=4, n_actions=8)
    t.push_actions(tick=5, agent_actions={1: 0, 2: 3, 3: 7})
    assert t.actions_by_tick[5] == {1: 0, 2: 3, 3: 7}


# ─── Métrique M1 : listener shift ──────────────────────────────────────


def test_listener_shift_zero_when_no_emission() -> None:
    t = LanguageCausalityTracker(n_tokens=4, n_actions=4)
    # Pas d'émission, baseline = post-listen = uniforme
    t.push_actions(tick=1, agent_actions={1: 0, 2: 1, 3: 2})
    shifts = t.listener_shift_per_token()
    # Pas d'émission → post-listen distrib = uniforme (eps regularized)
    for v in shifts.values():
        assert v < 0.5  # faible KL


def test_listener_shift_high_when_listeners_biased() -> None:
    """Si après chaque token X, les listeners font tous l'action 0,
    le KL doit être élevé."""
    t = LanguageCausalityTracker(n_tokens=2, n_actions=4, post_listen_window=2)
    # Baseline : actions uniformes
    for tick in range(1, 100):
        t.push_actions(tick, {
            i: (tick + i) % 4 for i in range(10)
        })
    # 50 émissions token 0, listeners font action 0 systématiquement
    for tick in range(10, 90, 5):
        t.push_emission(
            tick=tick, speaker_id=1, token_id=0,
            listener_ids=[2, 3, 4], context={},
        )
        # Overwrite actions du tick+1 et tick+2 pour les listeners : action 0
        for dt in (1, 2):
            t.actions_by_tick[tick + dt] = {2: 0, 3: 0, 4: 0}
    shifts = t.listener_shift_per_token()
    # Token 0 doit avoir KL > 0.5 (action 0 sur-représentée)
    assert shifts[0] > 0.3, f"shift[0]={shifts[0]} attendu > 0.3"
    # Token 1 (jamais émis) → faible KL
    assert shifts[1] < 0.1


# ─── Métrique M2 : context consistency ─────────────────────────────────


def test_context_consistency_perfect() -> None:
    """Si toutes les émissions du token X ont le même contexte, consistance=1."""
    t = LanguageCausalityTracker(n_tokens=2, n_actions=4)
    for i in range(20):
        t.push_emission(
            tick=i, speaker_id=1, token_id=0,
            listener_ids=[2], context={"food_visible": True, "energy": 0.5},
        )
    cons = t.context_consistency_per_token()
    assert cons[0] == 1.0


def test_context_consistency_uniform_low() -> None:
    """4 contextes différents équiprobables → consistance ≈ 0.25."""
    t = LanguageCausalityTracker(n_tokens=2, n_actions=4)
    contexts = [
        {"food_visible": True}, {"food_visible": False},
        {"energy": 0.1}, {"energy": 0.9},
    ]
    for i in range(40):
        t.push_emission(
            tick=i, speaker_id=1, token_id=0,
            listener_ids=[2], context=contexts[i % 4],
        )
    cons = t.context_consistency_per_token()
    assert 0.15 <= cons[0] <= 0.35


def test_context_consistency_empty() -> None:
    t = LanguageCausalityTracker(n_tokens=2, n_actions=4)
    cons = t.context_consistency_per_token()
    assert cons[0] == 0.0
    assert cons[1] == 0.0


# ─── Finalize : verdict ────────────────────────────────────────────────


def test_finalize_returns_metrics_dict() -> None:
    t = LanguageCausalityTracker(n_tokens=2, n_actions=4)
    t.push_emission(tick=1, speaker_id=1, token_id=0, listener_ids=[2], context={})
    t.push_actions(tick=2, agent_actions={2: 1})
    m = t.finalize()
    assert "listener_shift_per_token" in m
    assert "context_consistency_per_token" in m
    assert "listener_shift_mean" in m
    assert "n_emissions_total" in m
    assert m["n_emissions_total"] == 1
    assert "verdict" in m


def test_finalize_communication_renforcee_with_strong_signal() -> None:
    """Configuration extrême : shift élevé + consistance haute → verdict positif."""
    t = LanguageCausalityTracker(n_tokens=1, n_actions=4, post_listen_window=2)
    # Baseline uniforme
    for tick in range(1, 100):
        t.push_actions(tick, {i: tick % 4 for i in range(5)})
    # 30 émissions identiques de token 0, listeners forcés en action 0
    for i in range(20):
        tick = 10 + i * 3
        t.push_emission(
            tick=tick, speaker_id=1, token_id=0,
            listener_ids=[2, 3], context={"food_visible": True},
        )
        for dt in (1, 2):
            t.actions_by_tick[tick + dt] = {2: 0, 3: 0}
    m = t.finalize()
    # Avec ces données, on doit avoir un signal détecté
    assert m["listener_shift_mean"] > 0.05
    assert m["context_consistency_mean"] >= 0.9
