"""V8-B2.0 — Vocabulary émergent par lignée.

Chaque lignée (LineageBrain) détient sa propre `Vocabulary` :
    - N tokens, chacun = vecteur dense R^embedding_dim
    - Héritage 1:1 + mutation gaussienne à la reproduction
    - Pas de vocab partagé entre lignées (divergence linguistique)

Le langage n'est PAS donné — il émerge par sélection naturelle :
les lignées dont les agents coopèrent via vocalize (reward social
positif) survivent mieux et propagent leur vocabulary.

Spec : `docs/superpowers/specs/2026-05-24-aetherlife-v8-b2-emergent-language-design.md`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VocabularyConfig:
    """Configuration du langage émergent V8-B2.0."""

    enabled: bool = False
    n_tokens: int = 4              # vocab size par lignée (commence petit)
    embedding_dim: int = 16        # dim de chaque vecteur d'embedding
    mutation_std: float = 0.05     # bruit gaussien sur embeddings à héritage
    listen_radius: int = 5         # voisins audibles (Manhattan)
    init_std: float = 0.3          # std des embeddings initiaux
    embedding_clip_norm: float = 2.0   # cap norm L2 pour éviter explosion

    # V8-B2.0 — coût énergétique de vocalize (l'agent paie pour parler)
    # Émergence par sélection naturelle : si tokens utiles → lignée
    # survit → propagation. Si spam → lignée meurt. Pas de reward direct.
    vocalize_energy_cost: float = 0.05

    # DEPRECATED V8-B2.0 : on ne donne PAS de reward social direct (chat
    # artificiel). Conservé pour expérimentation A/B mais default = 0.
    social_bonus: float = 0.0
    social_window_ticks: int = 3

    # V8-B2.3 — Ablation interventionnelle.
    # Si != None, à partir de ce tick, les actions vocalize deviennent
    # no-op (pas d'émission, pas de coût énergétique). L'agent peut
    # toujours CHOISIR l'action, mais rien ne se passe. Test causal
    # gold standard : couper le canal et observer ce qui s'effondre.
    disable_vocalize_after_tick: int | None = None

    def __post_init__(self) -> None:
        if self.n_tokens <= 0:
            raise ValueError(f"n_tokens doit être > 0 (got {self.n_tokens})")
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim doit être > 0 (got {self.embedding_dim})"
            )
        if self.mutation_std < 0:
            raise ValueError(f"mutation_std doit être >= 0 (got {self.mutation_std})")
        if self.listen_radius < 0:
            raise ValueError(
                f"listen_radius doit être >= 0 (got {self.listen_radius})"
            )
        if self.init_std < 0:
            raise ValueError(f"init_std doit être >= 0 (got {self.init_std})")
        if self.embedding_clip_norm <= 0:
            raise ValueError(
                f"embedding_clip_norm doit être > 0 "
                f"(got {self.embedding_clip_norm})"
            )
        if self.social_bonus < 0:
            raise ValueError(f"social_bonus doit être >= 0 (got {self.social_bonus})")
        if self.social_window_ticks <= 0:
            raise ValueError(
                f"social_window_ticks doit être > 0 "
                f"(got {self.social_window_ticks})"
            )
        if self.vocalize_energy_cost < 0:
            raise ValueError(
                f"vocalize_energy_cost doit être >= 0 "
                f"(got {self.vocalize_energy_cost})"
            )
        if (self.disable_vocalize_after_tick is not None
                and self.disable_vocalize_after_tick < 0):
            raise ValueError(
                f"disable_vocalize_after_tick doit être >= 0 ou None "
                f"(got {self.disable_vocalize_after_tick})"
            )


class Vocabulary:
    """Dictionnaire de N tokens, chacun = embedding R^d.

    Tracking d'usage : `usage_count[i]` = nombre de fois que le token i
    a été vocalize par un agent de la lignée.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        cfg: VocabularyConfig,
    ) -> None:
        assert embeddings.shape == (cfg.n_tokens, cfg.embedding_dim)
        self.cfg = cfg
        self.embeddings = embeddings.astype(np.float32, copy=True)
        self.usage_count = np.zeros(cfg.n_tokens, dtype=np.int64)
        self._clip_embeddings()

    @classmethod
    def random(
        cls,
        cfg: VocabularyConfig,
        rng: np.random.Generator,
    ) -> "Vocabulary":
        """Init random gaussienne."""
        emb = rng.normal(0.0, cfg.init_std, size=(cfg.n_tokens, cfg.embedding_dim))
        return cls(embeddings=emb, cfg=cfg)

    def inherit(self, rng: np.random.Generator) -> "Vocabulary":
        """Clone + mutation gaussienne sur les embeddings.

        Le child n'hérite PAS de usage_count (reset à 0).
        """
        noise = rng.normal(0.0, self.cfg.mutation_std, size=self.embeddings.shape)
        new_emb = self.embeddings + noise
        return Vocabulary(embeddings=new_emb, cfg=self.cfg)

    def record_use(self, token_id: int) -> None:
        """Incrémente l'usage_count d'un token."""
        if 0 <= token_id < self.cfg.n_tokens:
            self.usage_count[token_id] += 1

    def get_embedding(self, token_id: int) -> np.ndarray:
        """Récupère l'embedding d'un token (copie pour éviter mutation externe)."""
        if not (0 <= token_id < self.cfg.n_tokens):
            return np.zeros(self.cfg.embedding_dim, dtype=np.float32)
        return self.embeddings[token_id].copy()

    def usage_entropy(self) -> float:
        """Entropie de la distribution d'usage. 0 = monopole, log(N) = uniforme.

        Mesure clé pour valider émergence : si entropie ~ log(n_tokens), tous
        les tokens sont utilisés également. Si entropie ~ 0, un seul token
        domine (vocabulary mort).
        """
        total = self.usage_count.sum()
        if total == 0:
            return 0.0
        p = self.usage_count.astype(np.float64) / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def distance_to(self, other: "Vocabulary") -> float:
        """Distance L2 entre deux vocabularies (somme des distances par token).

        Mesure clé de divergence linguistique inter-lignée.
        """
        if self.embeddings.shape != other.embeddings.shape:
            return float("inf")
        diff = self.embeddings - other.embeddings
        return float(np.sqrt(np.sum(diff * diff)))

    def _clip_embeddings(self) -> None:
        """Clip la norme L2 de chaque embedding pour éviter explosion."""
        for i in range(self.cfg.n_tokens):
            norm = np.linalg.norm(self.embeddings[i])
            if norm > self.cfg.embedding_clip_norm:
                self.embeddings[i] *= self.cfg.embedding_clip_norm / norm

    def __repr__(self) -> str:
        return (
            f"Vocabulary(n_tokens={self.cfg.n_tokens}, "
            f"embedding_dim={self.cfg.embedding_dim}, "
            f"total_usage={int(self.usage_count.sum())}, "
            f"entropy={self.usage_entropy():.2f})"
        )
