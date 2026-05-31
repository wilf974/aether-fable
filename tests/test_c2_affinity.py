import pytest

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
