import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import numpy as np


def test_build_env_default_n_seed_points_is_8():
    from overnight_v8b1 import build_env
    env = build_env(1, regime="coordination_collective")
    assert env.cfg.biomes.n_seed_points == 8


def test_build_env_propagates_n_seed_points():
    from overnight_v8b1 import build_env
    env = build_env(1, regime="coordination_collective", n_seed_points=16)
    assert env.cfg.biomes.n_seed_points == 16


def _boundaries(bm):
    # nb de paires de cellules adjacentes de biomes différents (fragmentation)
    h = int((bm[:, :-1] != bm[:, 1:]).sum())
    v = int((bm[:-1, :] != bm[1:, :]).sum())
    return h + v


def test_higher_n_seed_points_more_fragmented():
    from overnight_v8b1 import build_env
    e4 = build_env(1, regime="coordination_collective", n_seed_points=4)
    e4.reset(seed=1)
    e16 = build_env(1, regime="coordination_collective", n_seed_points=16)
    e16.reset(seed=1)
    assert _boundaries(e16.biome_map) > _boundaries(e4.biome_map)


def test_run_overnight_records_n_seed_points(tmp_path):
    from overnight_v8b1 import run_overnight
    rep = run_overnight(n_ticks=20, seed=1, device="cpu", out_dir=str(tmp_path),
                        regime="coordination_collective", n_seed_points=16)
    assert rep["config"]["n_seed_points"] == 16


def _fake_report(seed, k, n, n_alive, gather):
    return {
        "config": {"seed": seed, "n_initial_affinities": k, "n_seed_points": n},
        "final_state": {"n_alive": n_alive},
        "cooperative_v8c3": {"gather_successes_total": gather},
    }


def test_extract_topology_row():
    from aggregate_topology import extract_topo
    r = extract_topo(_fake_report(1, 1, 16, 0, 0))
    assert r["seed"] == 1 and r["k"] == 1 and r["n"] == 16
    assert r["extinct"] is True and r["n_alive"] == 0


def _row(seed, k, n, n_alive):
    from aggregate_topology import extract_topo
    return extract_topo(_fake_report(seed, k, n, n_alive, 50))


def test_summarize_topology_grid():
    from aggregate_topology import summarize_topo
    rows = [
        _row(1, 1, 4, 0), _row(2, 1, 4, 0),
        _row(1, 1, 16, 60), _row(2, 1, 16, 58),
        _row(1, 4, 4, 61), _row(1, 4, 16, 62),
    ]
    s = summarize_topo(rows)
    assert s["grid"][(1, 4)]["extinction_pct"] == 100
    assert s["grid"][(1, 16)]["extinction_pct"] == 0
    assert s["grid"][(4, 4)]["extinction_pct"] == 0
