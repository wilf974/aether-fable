"""Tests V8-B2.0 — Vocabulary émergent par lignée."""
from __future__ import annotations

import numpy as np
import pytest

from aetherlife.world.vocabulary import Vocabulary, VocabularyConfig


# ─── VocabularyConfig validation ────────────────────────────────────────


def test_vocab_config_defaults() -> None:
    cfg = VocabularyConfig()
    assert cfg.enabled is False
    assert cfg.n_tokens == 4
    assert cfg.embedding_dim == 16
    assert 0 < cfg.mutation_std < 1
    assert cfg.listen_radius >= 1
    assert cfg.embedding_clip_norm > 0
    # V8-B2.0 (révisé) : pas de reward social direct par défaut,
    # seulement coût énergétique vocalize (sélection au niveau lignée)
    assert cfg.social_bonus == 0.0
    assert cfg.vocalize_energy_cost > 0


def test_vocab_config_validates() -> None:
    with pytest.raises(ValueError):
        VocabularyConfig(n_tokens=0)
    with pytest.raises(ValueError):
        VocabularyConfig(embedding_dim=0)
    with pytest.raises(ValueError):
        VocabularyConfig(mutation_std=-0.1)
    with pytest.raises(ValueError):
        VocabularyConfig(listen_radius=-1)
    with pytest.raises(ValueError):
        VocabularyConfig(init_std=-0.1)
    with pytest.raises(ValueError):
        VocabularyConfig(embedding_clip_norm=0)
    with pytest.raises(ValueError):
        VocabularyConfig(social_bonus=-0.1)
    with pytest.raises(ValueError):
        VocabularyConfig(social_window_ticks=0)


# ─── Vocabulary creation ────────────────────────────────────────────────


def test_vocabulary_random_shape() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=5, embedding_dim=8)
    rng = np.random.default_rng(42)
    vocab = Vocabulary.random(cfg, rng)
    assert vocab.embeddings.shape == (5, 8)
    assert vocab.embeddings.dtype == np.float32
    assert vocab.usage_count.shape == (5,)
    assert vocab.usage_count.sum() == 0


def test_vocabulary_random_deterministic() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=3, embedding_dim=4)
    v1 = Vocabulary.random(cfg, np.random.default_rng(42))
    v2 = Vocabulary.random(cfg, np.random.default_rng(42))
    assert np.array_equal(v1.embeddings, v2.embeddings)


def test_vocabulary_random_different_seeds() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=3, embedding_dim=4)
    v1 = Vocabulary.random(cfg, np.random.default_rng(0))
    v2 = Vocabulary.random(cfg, np.random.default_rng(999))
    assert not np.array_equal(v1.embeddings, v2.embeddings)


# ─── Inheritance ────────────────────────────────────────────────────────


def test_vocabulary_inherit_zero_mutation_is_identity() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=8, mutation_std=0.0)
    parent = Vocabulary.random(cfg, np.random.default_rng(0))
    child = parent.inherit(np.random.default_rng(1))
    assert np.allclose(parent.embeddings, child.embeddings, atol=1e-6)


def test_vocabulary_inherit_diverges_with_mutation() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=8, mutation_std=0.1)
    parent = Vocabulary.random(cfg, np.random.default_rng(0))
    child = parent.inherit(np.random.default_rng(1))
    assert not np.allclose(parent.embeddings, child.embeddings, atol=1e-3)
    # Mais distance bornée
    diff = np.abs(parent.embeddings - child.embeddings).max()
    assert diff < 1.0  # < ~10 sigma


def test_vocabulary_inherit_resets_usage() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=4)
    parent = Vocabulary.random(cfg, np.random.default_rng(0))
    parent.record_use(0)
    parent.record_use(1)
    parent.record_use(1)
    assert parent.usage_count.sum() == 3
    child = parent.inherit(np.random.default_rng(1))
    assert child.usage_count.sum() == 0


# ─── Usage tracking ─────────────────────────────────────────────────────


def test_record_use_increments_count() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=4)
    vocab = Vocabulary.random(cfg, np.random.default_rng(0))
    vocab.record_use(2)
    vocab.record_use(2)
    vocab.record_use(0)
    assert vocab.usage_count[0] == 1
    assert vocab.usage_count[2] == 2


def test_record_use_invalid_token_ignored() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=4)
    vocab = Vocabulary.random(cfg, np.random.default_rng(0))
    vocab.record_use(99)  # OOB
    vocab.record_use(-1)
    assert vocab.usage_count.sum() == 0


def test_get_embedding_returns_vector() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=8)
    vocab = Vocabulary.random(cfg, np.random.default_rng(0))
    e = vocab.get_embedding(0)
    assert e.shape == (8,)


def test_get_embedding_invalid_returns_zeros() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=8)
    vocab = Vocabulary.random(cfg, np.random.default_rng(0))
    e = vocab.get_embedding(99)
    assert e.shape == (8,)
    assert np.all(e == 0)


# ─── Métriques d'émergence ──────────────────────────────────────────────


def test_usage_entropy_uniform() -> None:
    """Si tous les tokens sont utilisés également, entropie ~ log(N)."""
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=4)
    vocab = Vocabulary.random(cfg, np.random.default_rng(0))
    for tok in range(4):
        for _ in range(10):
            vocab.record_use(tok)
    expected = float(np.log(4))
    assert abs(vocab.usage_entropy() - expected) < 0.01


def test_usage_entropy_monopole() -> None:
    """Si un seul token domine, entropie ~ 0."""
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=4)
    vocab = Vocabulary.random(cfg, np.random.default_rng(0))
    for _ in range(100):
        vocab.record_use(0)
    assert vocab.usage_entropy() == pytest.approx(0.0)


def test_usage_entropy_empty() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=4)
    vocab = Vocabulary.random(cfg, np.random.default_rng(0))
    assert vocab.usage_entropy() == 0.0


def test_vocabulary_distance() -> None:
    cfg = VocabularyConfig(enabled=True, n_tokens=4, embedding_dim=4)
    v1 = Vocabulary.random(cfg, np.random.default_rng(0))
    v2 = Vocabulary.random(cfg, np.random.default_rng(999))
    d = v1.distance_to(v2)
    assert d > 0  # vocabularies différents
    # Self-distance = 0
    assert v1.distance_to(v1) == pytest.approx(0.0)


def test_embedding_clip_norm() -> None:
    """Les embeddings doivent être clippés à norme <= embedding_clip_norm."""
    cfg = VocabularyConfig(
        enabled=True, n_tokens=2, embedding_dim=4,
        embedding_clip_norm=1.0, init_std=10.0,  # init énorme pour forcer clip
    )
    vocab = Vocabulary.random(cfg, np.random.default_rng(0))
    for i in range(2):
        norm = np.linalg.norm(vocab.embeddings[i])
        assert norm <= 1.0 + 1e-5
