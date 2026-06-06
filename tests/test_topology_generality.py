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
