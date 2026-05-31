import os
import sys

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


def test_run_overnight_records_condition_in_report(tmp_path):
    from overnight_v8b1 import run_overnight
    report = run_overnight(
        n_ticks=20, seed=1, device="cpu", out_dir=str(tmp_path),
        regime="coordination_collective", n_initial_affinities=2,
    )
    assert report["config"]["n_initial_affinities"] == 2
