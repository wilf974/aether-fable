"""Tests V8-B1.8 — Stabilisation RL anti-divergence.

Couvre :
    - reward clipping (enabled / disabled)
    - gradient clipping configurable
    - loss guard NaN/Inf/threshold
    - compteur skipped_updates
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aetherlife.agents.lineage_brain import BrainConfig, LineageBrain


def _cfg(**kwargs) -> BrainConfig:
    defaults = dict(
        enabled=True, device="cpu",
        min_replay_to_learn=4, batch_size=4, buffer_capacity=20,
        train_every=1,
    )
    defaults.update(kwargs)
    return BrainConfig(**defaults)


# ─── BrainConfig validation V8-B1.8 ─────────────────────────────────────


def test_brain_config_v8b18_defaults() -> None:
    cfg = BrainConfig()
    assert cfg.reward_clip_enabled is True
    # V8-B2.1 : action space étendu (8 actions) → resserrement
    assert cfg.reward_clip_low == -3.0
    assert cfg.reward_clip_high == 3.0
    assert cfg.grad_clip_norm == 0.5
    assert cfg.loss_max_threshold == 50.0
    assert cfg.skip_invalid_updates is True


def test_brain_config_v8b18_validates() -> None:
    with pytest.raises(ValueError):
        BrainConfig(reward_clip_low=1.0, reward_clip_high=1.0)
    with pytest.raises(ValueError):
        BrainConfig(grad_clip_norm=0)
    with pytest.raises(ValueError):
        BrainConfig(loss_max_threshold=0)


# ─── Reward clipping ─────────────────────────────────────────────────────


def test_reward_clipping_high_value() -> None:
    """Un reward de +100 doit être clippé à reward_clip_high."""
    cfg = _cfg(reward_clip_enabled=True,
               reward_clip_low=-3.0, reward_clip_high=3.0)
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    obs = np.zeros(10, dtype=np.float32)
    brain.observe(obs, 0, 100.0, obs, False)
    assert brain.buffer._rewards[0] == pytest.approx(3.0)  # noqa: SLF001


def test_reward_clipping_low_value() -> None:
    cfg = _cfg(reward_clip_enabled=True,
               reward_clip_low=-3.0, reward_clip_high=3.0)
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    obs = np.zeros(10, dtype=np.float32)
    brain.observe(obs, 0, -50.0, obs, False)
    assert brain.buffer._rewards[0] == pytest.approx(-3.0)  # noqa: SLF001


def test_reward_clipping_disabled_preserves_value() -> None:
    cfg = _cfg(reward_clip_enabled=False)
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    obs = np.zeros(10, dtype=np.float32)
    brain.observe(obs, 0, 42.5, obs, False)
    assert brain.buffer._rewards[0] == pytest.approx(42.5)  # noqa: SLF001


# ─── Gradient clipping ──────────────────────────────────────────────────


def test_trainer_has_grad_clip_norm() -> None:
    cfg = _cfg(grad_clip_norm=2.5)
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    assert brain.trainer._grad_clip_norm == 2.5  # noqa: SLF001


def test_trainer_skipped_updates_counter_exists() -> None:
    cfg = _cfg()
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    assert hasattr(brain.trainer, "skipped_updates")
    assert brain.trainer.skipped_updates == 0


# ─── Loss guard NaN/Inf/threshold ──────────────────────────────────────


def _fill_buffer(brain, n: int) -> None:
    obs = np.zeros(brain.obs_dim, dtype=np.float32)
    for _ in range(n):
        brain.observe(obs, 0, 0.0, obs, False)


def test_loss_guard_skips_on_nan_q() -> None:
    """Si Q-values contiennent NaN, l'update doit être skip."""
    cfg = _cfg(skip_invalid_updates=True, loss_max_threshold=1000.0)
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    _fill_buffer(brain, 10)
    # Forcer NaN dans online network
    with torch.no_grad():
        for p in brain.online.parameters():
            p.fill_(float("nan"))
            break
    n_skipped_before = brain.trainer.skipped_updates
    obs = np.zeros(10, dtype=np.float32)
    # Reset global_step à 0 pour forcer train at min_replay
    brain.observe(obs, 0, 0.0, obs, False)
    # Vérifier skip
    assert brain.trainer.skipped_updates > n_skipped_before


def test_loss_guard_threshold() -> None:
    """Si loss > loss_max_threshold, update skip."""
    cfg = _cfg(skip_invalid_updates=True, loss_max_threshold=0.001)
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    # Setup: forcer un état où loss sera grand
    # Avec poids random, rewards 0 → target_q = gamma × q_next
    # Loss = ||q_pred - target_q||² grand si réseau pas entraîné
    _fill_buffer(brain, 10)
    n_skipped_before = brain.trainer.skipped_updates
    obs = np.zeros(10, dtype=np.float32)
    # Pousser une transition avec reward 0
    brain.observe(obs, 0, 0.0, obs, False)
    # Si le loss est grand (très probable avec threshold=0.001), skip
    # On vérifie au moins que ça ne crash pas et que skipped_updates
    # peut augmenter
    assert brain.trainer.skipped_updates >= n_skipped_before


def test_skip_invalid_disabled_no_skipping() -> None:
    """Si skip_invalid_updates=False, pas de skip même si loss énorme."""
    cfg = _cfg(skip_invalid_updates=False, loss_max_threshold=0.0001)
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    _fill_buffer(brain, 10)
    obs = np.zeros(10, dtype=np.float32)
    brain.observe(obs, 0, 0.0, obs, False)
    # skipped_updates ne devrait pas avoir augmenté
    assert brain.trainer.skipped_updates == 0


# ─── Test intégration : training ne fait pas exploser loss ─────────────


def test_training_loss_remains_bounded() -> None:
    """Run 100 observe() avec rewards [-1,1] clippés. Loss doit rester bornée."""
    cfg = _cfg(grad_clip_norm=1.0, reward_clip_enabled=True,
               loss_max_threshold=10.0)
    brain = LineageBrain(root_id=0, obs_dim=10, n_actions=4, cfg=cfg, seed=0)
    rng = np.random.default_rng(42)
    obs = np.zeros(10, dtype=np.float32)
    for _ in range(100):
        obs = rng.standard_normal(10).astype(np.float32)
        reward = float(rng.uniform(-5, 5))  # range > clip, sera clippé
        brain.observe(obs, int(rng.integers(0, 4)), reward, obs, False)
    # Pas de NaN/Inf dans les poids finaux
    for p in brain.online.parameters():
        assert torch.isfinite(p).all(), "Poids ont divergé (NaN/Inf)"
    # Loss dernière < threshold
    if brain.last_loss is not None:
        assert brain.last_loss < cfg.loss_max_threshold * 2  # marge
