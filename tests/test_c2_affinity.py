import os
import sys
from collections import Counter

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

from aetherlife.world.biomes import BiomeConfig


def test_n_initial_affinities_default_is_4():
    assert BiomeConfig().n_initial_affinities == 4


def test_n_initial_affinities_accepts_1_2_4():
    for k in (1, 2, 4):
        assert BiomeConfig(n_initial_affinities=k).n_initial_affinities == k


def test_n_initial_affinities_rejects_zero():
    with pytest.raises(ValueError):
        BiomeConfig(n_initial_affinities=0)


def test_n_initial_affinities_rejects_above_4():
    with pytest.raises(ValueError):
        BiomeConfig(n_initial_affinities=5)


def test_build_env_propagates_n_initial_affinities():
    from overnight_v8b1 import build_env
    env = build_env(seed=1, regime="coordination_collective",
                    n_initial_affinities=2)
    assert env.cfg.biomes.n_initial_affinities == 2


def test_build_env_defaults_to_4():
    from overnight_v8b1 import build_env
    env = build_env(seed=1, regime="coordination_collective")
    assert env.cfg.biomes.n_initial_affinities == 4


def test_build_env_rejects_non_default_affinities_outside_coordination():
    from overnight_v8b1 import build_env
    with pytest.raises(ValueError):
        build_env(seed=1, regime="training", n_initial_affinities=2)


def test_run_overnight_records_condition_in_report(tmp_path):
    pytest.importorskip("torch", reason="suite complete : requiert torch")
    pytest.importorskip("mw_ia", reason="suite complete : requiert le repo sibling MW_IA")
    from overnight_v8b1 import run_overnight
    report = run_overnight(
        n_ticks=20, seed=1, device="cpu", out_dir=str(tmp_path),
        regime="coordination_collective", n_initial_affinities=2,
    )
    assert report["config"]["n_initial_affinities"] == 2


def _affinities(k):
    from overnight_v8b1 import build_env
    env = build_env(seed=1, regime="coordination_collective",
                    n_initial_affinities=k)
    env.reset(seed=1)
    return Counter(a.biome_affinity for a in env._agents)  # noqa: SLF001


def test_reset_k1_all_affinity_zero():
    assert set(_affinities(1)) == {0}


def test_reset_k2_two_affinities_balanced():
    c = _affinities(2)
    assert set(c) == {0, 1}
    assert c[0] == 10 and c[1] == 10  # 20 agents, round-robin %2


def test_reset_k4_balanced_5_each_nonregression():
    assert dict(_affinities(4)) == {0: 5, 1: 5, 2: 5, 3: 5}


def _fake_report(seed, k, mobility, n_alive, gather, aff_dist):
    return {
        "config": {"seed": seed, "n_initial_affinities": k},
        "spatial_mobility_v8c3": {
            "corr_occupation_start_end": mobility,
            "village_basin": (mobility is not None and mobility >= 0.8),
        },
        "final_state": {"n_alive": n_alive, "affinity_distribution": aff_dist},
        "cooperative_v8c3": {"gather_successes_total": gather},
    }


def test_extract_c2_row():
    from aggregate_c2 import extract_c2
    r = extract_c2(_fake_report(1, 1, 0.9, 60, 120, {"0": 60}))
    assert r["seed"] == 1 and r["k"] == 1
    assert r["mobility_score"] == 0.9 and r["village_basin"] is True
    assert r["n_alive"] == 60 and r["gather_successes"] == 120
    assert r["extinction"] is False
    assert r["aff_conc_final"] == 1.0  # 60/60


def test_extract_c2_extinction_and_affconc():
    from aggregate_c2 import extract_c2
    r = extract_c2(_fake_report(2, 4, None, 0, 0, {"0": 0}))
    assert r["extinction"] is True
    assert r["aff_conc_final"] == 0.0  # population vide


def test_summarize_c2_paired_delta_and_sign():
    from aggregate_c2 import summarize_c2
    rows = [
        extract_dict(1, 1, 0.90), extract_dict(1, 4, 0.40),
        extract_dict(2, 1, 0.85), extract_dict(2, 4, 0.50),
        extract_dict(3, 1, 0.30), extract_dict(3, 4, 0.60),  # contre-exemple
    ]
    summary = summarize_c2(rows)
    # delta intra-seed k1-k4
    assert summary["paired"][1]["delta_k1_k4"] == pytest.approx(0.50)
    # 2/3 seeds ont k1 > k4
    assert summary["n_seeds_k1_gt_k4"] == 2
    assert summary["n_paired"] == 3
    # mobility_k2 presente dans paired (None si k2 absent du jeu de donnees)
    assert "mobility_k2" in summary["paired"][1]
    assert summary["paired"][1]["mobility_k2"] is None


def extract_dict(seed, k, mobility):
    from aggregate_c2 import extract_c2
    return extract_c2(_fake_report(seed, k, mobility, 60, 100, {"0": 50, "1": 10}))
