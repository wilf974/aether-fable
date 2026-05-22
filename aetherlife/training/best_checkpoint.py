"""BestCheckpointTracker — port léger du sous-projet V2-V de MW_IA.

Surveille un score d'évaluation périodique, sauvegarde le modèle au pic,
détecte stagnation pour early stopping, et expose un rollback.

Le score est typiquement un winrate eval greedy (∈ [0, 1]) mais peut être
n'importe quelle quantité où plus est mieux.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class _Savable(Protocol):
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...


@dataclass
class BestCheckpointTracker:
    """Suivi du best score d'eval et persistance du modèle au pic.

    Args:
        save_path: chemin du checkpoint best.
        patience: nombre d'évals consécutives sans amélioration avant `should_stop=True`.
        min_delta: amélioration minimale pour considérer un nouveau best.
    """

    save_path: Path
    patience: int = 10
    min_delta: float = 0.001

    best_score: float = float("-inf")
    best_step: int = 0
    evals_since_best: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.save_path = Path(self.save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def report(self, step: int, score: float, model: _Savable) -> bool:
        """Enregistre un score d'eval. Retourne True si nouveau best (et modèle sauvé).

        Sauvegarde le modèle dans `save_path` quand `score > best_score + min_delta`.
        Sinon incrémente `evals_since_best`.
        """
        self.history.append((step, score))
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.best_step = step
            self.evals_since_best = 0
            model.save(self.save_path)
        else:
            self.evals_since_best += 1
        return improved

    @property
    def should_stop(self) -> bool:
        """True si patience écoulée."""
        return self.evals_since_best >= self.patience

    def rollback(self, model: _Savable) -> None:
        """Restaure le modèle au best checkpoint."""
        if not self.save_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {self.save_path} — appeler report() au moins une fois."
            )
        model.load(self.save_path)
