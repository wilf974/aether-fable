"""Tests V8-B1 — LineageBrain : RL héritable par lignée.

Couvre :
    - BrainConfig validation
    - LineageBrain construction (obs_dim, n_actions ad hoc)
    - act() retourne action valide
    - observe() pousse dans le buffer
    - inherit_from() : poids enfant proches du parent, MSE bornée
    - inherit_from(mutation_std=0) : enfant === parent
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aetherlife.agents.lineage_brain import BrainConfig, LineageBrain


# ─── BrainConfig validation ───────────────────────────────────────────────


def test_brain_config_defaults() -> None:
    cfg = BrainConfig()
    assert cfg.enabled is False
    assert cfg.hidden_dims == (64, 64)
    assert 0 < cfg.lr < 1
    assert cfg.mutation_std >= 0
    assert cfg.vision_radius >= 1


def test_brain_config_validates() -> None:
    with pytest.raises(ValueError):
        BrainConfig(lr=0)
    with pytest.raises(ValueError):
        BrainConfig(gamma=-0.1)
    with pytest.raises(ValueError):
        BrainConfig(batch_size=0)
    with pytest.raises(ValueError):
        BrainConfig(buffer_capacity=0)
    with pytest.raises(ValueError):
        BrainConfig(mutation_std=-0.1)
    with pytest.raises(ValueError):
        BrainConfig(vision_radius=0)
    with pytest.raises(ValueError):
        BrainConfig(epsilon_start=-0.1)
    with pytest.raises(ValueError):
        BrainConfig(epsilon_end=1.5)


# ─── LineageBrain construction ────────────────────────────────────────────


def test_brain_construction_random_init() -> None:
    cfg = BrainConfig(enabled=True, device="cpu")
    brain = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    assert brain.root_id == 0
    assert brain.obs_dim == 10
    assert brain.n_actions == 4
    assert brain.global_step == 0


def test_brain_act_returns_valid_action() -> None:
    cfg = BrainConfig(enabled=True, device="cpu")
    brain = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    obs = np.zeros(10, dtype=np.float32)
    a = brain.act(obs, greedy=True)
    assert 0 <= a < 4


def test_brain_act_epsilon_explore() -> None:
    """Avec epsilon=1.0, l'agent explore (action peut être n'importe quoi)."""
    cfg = BrainConfig(
        enabled=True, device="cpu",
        epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=1,
    )
    brain = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=42,
    )
    obs = np.zeros(10, dtype=np.float32)
    actions = {brain.act(obs, greedy=False) for _ in range(100)}
    # On veut au moins 2 actions différentes pour confirmer l'exploration
    assert len(actions) >= 2


def test_brain_observe_pushes_to_buffer() -> None:
    cfg = BrainConfig(enabled=True, device="cpu")
    brain = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    obs = np.zeros(10, dtype=np.float32)
    next_obs = np.ones(10, dtype=np.float32)
    for _ in range(5):
        brain.observe(obs, 0, 1.0, next_obs, False)
    assert brain.global_step == 5
    assert len(brain.buffer) == 5


# ─── Héritage avec mutation ───────────────────────────────────────────────


def test_brain_inherit_zero_mutation_is_identity() -> None:
    """mutation_std=0 → enfant a EXACTEMENT les mêmes poids que le parent."""
    cfg = BrainConfig(enabled=True, device="cpu", mutation_std=0.0)
    parent = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    child = LineageBrain.inherit_from(
        parent=parent, root_id=1, mutation_std=0.0, seed=1,
    )
    parent_params = list(parent.online.parameters())
    child_params = list(child.online.parameters())
    assert len(parent_params) == len(child_params)
    for p, c in zip(parent_params, child_params):
        assert torch.allclose(p, c, atol=1e-9), (
            "mutation_std=0 doit donner enfant identique au parent"
        )


def test_brain_inherit_mutation_changes_weights() -> None:
    """mutation_std>0 → poids enfant diffèrent du parent."""
    cfg = BrainConfig(enabled=True, device="cpu")
    parent = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    child = LineageBrain.inherit_from(
        parent=parent, root_id=1, mutation_std=0.1, seed=42,
    )
    parent_p = next(parent.online.parameters())
    child_p = next(child.online.parameters())
    assert not torch.allclose(parent_p, child_p, atol=1e-6), (
        "mutation_std=0.1 doit perturber les poids"
    )
    # Mais la différence doit rester bornée
    diff = (child_p - parent_p).abs().mean().item()
    assert diff < 0.5, f"diff trop grande après mutation: {diff}"


def test_brain_inherit_preserves_architecture() -> None:
    """Enfant a même architecture (input/output/hidden) que le parent."""
    cfg = BrainConfig(enabled=True, device="cpu", hidden_dims=(32, 16))
    parent = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    child = LineageBrain.inherit_from(
        parent=parent, root_id=1, mutation_std=0.05, seed=1,
    )
    assert child.obs_dim == parent.obs_dim
    assert child.n_actions == parent.n_actions
    # Sanity check : peut agir
    obs = np.zeros(10, dtype=np.float32)
    a = child.act(obs, greedy=True)
    assert 0 <= a < 4


def test_brain_inherit_resets_global_step() -> None:
    """Enfant démarre à global_step=0 (pas hérité)."""
    cfg = BrainConfig(enabled=True, device="cpu")
    parent = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    # Faire avancer le parent
    obs = np.zeros(10, dtype=np.float32)
    for _ in range(10):
        parent.observe(obs, 0, 0.5, obs, False)
    assert parent.global_step == 10
    child = LineageBrain.inherit_from(parent=parent, root_id=1, seed=1)
    assert child.global_step == 0


def test_brain_inherit_independent_buffer() -> None:
    """Enfant a son propre buffer vide (pas partagé avec parent)."""
    cfg = BrainConfig(enabled=True, device="cpu")
    parent = LineageBrain(
        root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    obs = np.zeros(10, dtype=np.float32)
    for _ in range(5):
        parent.observe(obs, 0, 1.0, obs, False)
    child = LineageBrain.inherit_from(parent=parent, root_id=1, seed=1)
    assert len(parent.buffer) == 5
    assert len(child.buffer) == 0  # buffer enfant vide


def test_brain_root_id_stored() -> None:
    cfg = BrainConfig(enabled=True, device="cpu")
    brain = LineageBrain(
        root_id=42, obs_dim=10, n_actions=4, cfg=cfg, seed=0,
    )
    assert brain.root_id == 42
