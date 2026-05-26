"""Tests V8-C3 — CooperativeMetricsTracker."""
from __future__ import annotations

from aetherlife.world.cooperative_metrics import (
    CooperativeMetricsConfig,
    CooperativeMetricsTracker,
)


def test_empty_tracker_finalizes() -> None:
    tr = CooperativeMetricsTracker()
    out = tr.finalize()
    assert out["n_successes_observed"] == 0
    assert out["clustering_pre_success"]["n"] == 0
    assert out["success_chains"]["n_chains"] == 0
    assert out["token_entropy_pre_success"]["dominant_token"] is None


def test_clustering_records_neighbor_count() -> None:
    tr = CooperativeMetricsTracker()
    # Spot @ (5,5). 3 agents dans rayon 3 (manhattan)
    alive = [(5, 5), (5, 6), (6, 5), (10, 10)]
    tr.track_success(
        tick=100, pos=(5, 5), participants=[0, 1],
        all_alive_positions=alive,
    )
    out = tr.finalize()
    assert out["clustering_pre_success"]["n"] == 1
    # (5,5), (5,6), (6,5) sont à distance manhattan ≤ 3 du spot
    # (10,10) est à 10 → exclu
    assert out["clustering_pre_success"]["mean_neighbors_r3"] == 3


def test_vocalize_delay_measured() -> None:
    tr = CooperativeMetricsTracker()
    # Agent 0 vocalize token 1 à t=95, succès à t=100 → delay=5
    tr.track_vocalize(tick=95, agent_id=0, token_id=1, pos=(5, 5))
    tr.track_vocalize(tick=98, agent_id=1, token_id=2, pos=(5, 6))
    tr.track_success(
        tick=100, pos=(5, 5), participants=[0, 1],
        all_alive_positions=[(5, 5), (5, 6)],
    )
    out = tr.finalize()
    delay = out["vocalize_to_gather_delay"]
    assert delay["n_with_token"] == 1
    # min delay = min(100-95, 100-98) = 2
    assert delay["mean_min_delay"] == 2


def test_token_dominance_detected() -> None:
    tr = CooperativeMetricsTracker()
    # 3 succès, token 2 dominant
    for tick in [100, 110, 120]:
        tr.track_vocalize(tick=tick - 1, agent_id=0, token_id=2, pos=(5, 5))
        tr.track_vocalize(tick=tick - 1, agent_id=1, token_id=2, pos=(5, 6))
        tr.track_success(
            tick=tick, pos=(5, 5), participants=[0, 1],
            all_alive_positions=[(5, 5), (5, 6)],
        )
    out = tr.finalize()
    tok = out["token_entropy_pre_success"]
    assert tok["dominant_token"] == 2
    assert tok["dominant_share"] == 1.0
    assert tok["entropy"] == 0.0  # token unique → entropie nulle


def test_success_chains_detected() -> None:
    tr = CooperativeMetricsTracker(
        CooperativeMetricsConfig(chain_window=10),
    )
    # Chain 1 : 3 succès rapprochés (ticks 100, 105, 108)
    for t in [100, 105, 108]:
        tr.track_success(
            tick=t, pos=(5, 5), participants=[0, 1],
            all_alive_positions=[(5, 5), (5, 6)],
        )
    # Chain 2 (gap) : ticks 200, 203
    for t in [200, 203]:
        tr.track_success(
            tick=t, pos=(5, 5), participants=[0, 1],
            all_alive_positions=[(5, 5), (5, 6)],
        )
    out = tr.finalize()
    chains = out["success_chains"]
    assert chains["n_chains"] == 2
    assert chains["max_chain_len"] == 3
    assert chains["n_cascade_successes"] == 3  # chain de 3


def test_prune_old_vocalize_bounds_memory() -> None:
    tr = CooperativeMetricsTracker(
        CooperativeMetricsConfig(pre_success_window=5),
    )
    for t in range(0, 1000):
        tr.track_vocalize(tick=t, agent_id=0, token_id=0, pos=(0, 0))
    tr.prune_old_vocalize(current_tick=1000)
    # Après prune, il doit rester au plus W+1 entrées
    assert len(tr._vocalize_log) <= 6


def test_delay_trend_negative_indicates_learning() -> None:
    """Si les délais diminuent (apprentissage), trend_q4-q1 doit être < 0."""
    tr = CooperativeMetricsTracker(
        CooperativeMetricsConfig(pre_success_window=10),
    )
    # 8 succès. Délais : 10, 9, 8, 7, 4, 3, 2, 1 (apprentissage)
    delays = [10, 9, 8, 7, 4, 3, 2, 1]
    for i, dt in enumerate(delays):
        succ_tick = 1000 + i * 100  # bien espacés
        tr.track_vocalize(
            tick=succ_tick - dt, agent_id=0, token_id=0, pos=(0, 0),
        )
        tr.track_success(
            tick=succ_tick, pos=(0, 0), participants=[0],
            all_alive_positions=[(0, 0)],
        )
    out = tr.finalize()
    trend = out["vocalize_to_gather_delay"]["trend_q4_minus_q1"]
    assert trend < 0  # délai diminue → apprentissage
