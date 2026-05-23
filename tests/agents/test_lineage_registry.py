"""Tests V8-B1.1 — LineageRegistry : index des cerveaux par root_ancestor_id."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aetherlife.agents.lineage_brain import BrainConfig, LineageBrain
from aetherlife.agents.lineage_registry import LineageRegistry


def _cfg() -> BrainConfig:
    return BrainConfig(enabled=True, device="cpu")


def test_registry_empty_init() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    assert len(reg) == 0
    assert reg.alive_roots() == set()


def test_registry_get_or_create_new() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    b0 = reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    assert isinstance(b0, LineageBrain)
    assert b0.root_id == 0
    assert len(reg) == 1


def test_registry_returns_same_brain_on_second_call() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    b0a = reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    b0b = reg.get_or_create(root_id=0, parent_brain=None, seed=99)
    assert b0a is b0b  # identité, pas nouvelle instance


def test_registry_two_lineages_independent() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    b0 = reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    b1 = reg.get_or_create(root_id=1, parent_brain=None, seed=1)
    assert b0 is not b1
    assert len(reg) == 2


def test_registry_inherit_via_parent_brain() -> None:
    """Si parent_brain fourni, le nouveau brain hérite avec mutation."""
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    parent = reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    child = reg.get_or_create(root_id=1, parent_brain=parent, seed=1)
    # Poids enfant proches du parent (mutation = 0.02 par défaut)
    p_params = list(parent.online.parameters())
    c_params = list(child.online.parameters())
    max_diff = max(
        (c - p).abs().max().item() for p, c in zip(p_params, c_params)
    )
    assert 0 < max_diff < 0.5


def test_registry_cull_dead_lineages() -> None:
    """Cull supprime les cerveaux dont la lignée est éteinte."""
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    reg.get_or_create(root_id=1, parent_brain=None, seed=1)
    reg.get_or_create(root_id=2, parent_brain=None, seed=2)
    assert len(reg) == 3
    reg.cull_dead_lineages(alive_roots={0, 2})
    assert len(reg) == 2
    assert 0 in reg
    assert 1 not in reg
    assert 2 in reg


def test_registry_contains() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    reg.get_or_create(root_id=42, parent_brain=None, seed=0)
    assert 42 in reg
    assert 99 not in reg


def test_registry_iter() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    reg.get_or_create(root_id=1, parent_brain=None, seed=1)
    roots = set(reg.alive_roots())
    assert roots == {0, 1}


def test_registry_total_global_steps() -> None:
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    b0 = reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    b1 = reg.get_or_create(root_id=1, parent_brain=None, seed=1)
    obs = np.zeros(10, dtype=np.float32)
    for _ in range(5):
        b0.observe(obs, 0, 0.1, obs, False)
    for _ in range(3):
        b1.observe(obs, 1, 0.2, obs, False)
    assert reg.total_global_steps() == 8


def test_registry_act_via_lookup() -> None:
    """act_for_agent(agent) doit retourner action via le brain de sa lignée."""
    reg = LineageRegistry(cfg=_cfg(), obs_dim=10, n_actions=4)
    reg.get_or_create(root_id=0, parent_brain=None, seed=0)
    obs = np.zeros(10, dtype=np.float32)
    a = reg.act_for_lineage(root_id=0, obs=obs, greedy=True)
    assert 0 <= a < 4
