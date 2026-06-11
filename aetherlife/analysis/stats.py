"""Primitives statistiques pour l'agrégation multi-seeds (V2.5).

Pur stdlib/math — aucune dépendance (le projet n'a pas scipy). Conçu pour les
tailles d'échantillon des préenregistrements AetherLife (N ~ 10–100).

Contenu :
- ``wilson_ci`` : IC de Wilson pour une proportion (utilisé dans les préreg).
- ``bootstrap_ci`` : IC bootstrap percentile d'une moyenne (déterministe via seed).
- ``summarize`` : n / mean / std / sem / IC d'un échantillon numérique.
- ``proportion`` : fraction d'éléments satisfaisant un prédicat (+ IC Wilson).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

__all__ = ["wilson_ci", "bootstrap_ci", "summarize", "proportion", "Summary", "Proportion"]

# z critique pour les niveaux de confiance usuels (évite scipy)
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def _z_for(confidence: float) -> float:
    if confidence in _Z:
        return _Z[confidence]
    raise ValueError(f"confidence doit être dans {sorted(_Z)} (got {confidence})")


@dataclass(frozen=True)
class Proportion:
    """Proportion observée + IC de Wilson."""

    successes: int
    n: int
    p: float
    lo: float
    hi: float
    confidence: float


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> Proportion:
    """IC de Wilson pour une proportion (robuste aux petits n et p extrêmes).

    Args:
        successes: nombre de succès (0 ≤ successes ≤ n).
        n: taille de l'échantillon (≥ 0).
        confidence: niveau (0.90, 0.95 ou 0.99).

    Returns:
        Proportion(p, lo, hi). n=0 → p=lo=hi=0.0.
    """
    if not (0 <= successes <= n):
        raise ValueError(f"successes={successes} hors [0, n={n}]")
    if n == 0:
        return Proportion(0, 0, 0.0, 0.0, 0.0, confidence)
    z = _z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return Proportion(successes, n, p, max(0.0, center - half), min(1.0, center + half), confidence)


@dataclass(frozen=True)
class Summary:
    """Résumé d'un échantillon numérique."""

    n: int
    mean: float
    std: float            # écart-type d'échantillon (ddof=1) ; 0.0 si n<2
    sem: float            # erreur standard de la moyenne
    ci_lo: float
    ci_hi: float
    confidence: float
    method: str           # "bootstrap" | "normal" | "degenerate"


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """IC percentile bootstrap de la moyenne (déterministe via ``seed``)."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    alpha = 1.0 - confidence
    lo = means[max(0, int((alpha / 2) * n_resamples))]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


def summarize(
    values: Iterable[float],
    confidence: float = 0.95,
    *,
    method: str = "bootstrap",
    seed: int = 0,
) -> Summary:
    """Résume un échantillon : n, moyenne, écart-type, SEM, IC.

    method="bootstrap" (défaut, robuste) ou "normal" (IC = mean ± z·sem).
    """
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, confidence, "degenerate")
    mean = _mean(xs)
    if n == 1:
        return Summary(1, mean, 0.0, 0.0, mean, mean, confidence, "degenerate")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    if method == "normal":
        z = _z_for(confidence)
        lo, hi = mean - z * sem, mean + z * sem
    elif method == "bootstrap":
        lo, hi = bootstrap_ci(xs, confidence, seed=seed)
    else:
        raise ValueError(f"method inconnue : {method!r}")
    return Summary(n, mean, std, sem, lo, hi, confidence, method)


def proportion(
    values: Iterable,
    predicate: Callable[[object], bool] = bool,
    confidence: float = 0.95,
) -> Proportion:
    """Fraction d'éléments satisfaisant ``predicate`` + IC de Wilson."""
    items = list(values)
    successes = sum(1 for v in items if predicate(v))
    return wilson_ci(successes, len(items), confidence)
