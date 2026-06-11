"""Métriques d'écologie synthétique (V2.5).

Quantifie la structure écologique d'une population d'agents au-delà des
simples taux de survie : diversité, dominance, recouvrement de niche
spatiale entre stratégies (affinités), et détection naïve de bifurcation
sur une série temporelle.

Conçu comme observateur pur : ne consomme que des positions/affinités déjà
produites (events v8 ou appels directs), n'influence jamais la dynamique.

Toutes les fonctions sont numpy/stdlib only — aucune dépendance optionnelle.

Références :
    - Shannon H' = -Σ p_i ln(p_i)        (diversité)
    - Simpson  λ = Σ p_i²                 (dominance ; D = 1-λ = diversité)
    - Pianka   O_jk = Σ p_ij p_ik / sqrt(Σ p_ij² · Σ p_ik²)   (recouvrement de niche)
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "shannon_diversity",
    "simpson_dominance",
    "pianka_overlap",
    "detect_bifurcation",
    "EcologyTracker",
]


def shannon_diversity(counts: Iterable[float], *, normalized: bool = False) -> float:
    """Indice de Shannon H' = -Σ p_i ln(p_i).

    Args:
        counts: effectifs (ou proportions) par catégorie. Les zéros sont ignorés.
        normalized: si True, divise par ln(S) (S = nb catégories non vides),
            ramenant H' dans [0, 1] (évenness de Pielou). 0 catégorie utile → 0.

    Returns:
        H' ≥ 0. Une seule catégorie peuplée → 0 (diversité nulle).
    """
    vals = [float(c) for c in counts if c > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    h = -sum((v / total) * math.log(v / total) for v in vals)
    if normalized:
        s = len(vals)
        return h / math.log(s) if s > 1 else 0.0
    return h


def simpson_dominance(counts: Iterable[float]) -> float:
    """Indice de dominance de Simpson λ = Σ p_i².

    Returns:
        λ ∈ (0, 1]. Proche de 1 = une catégorie domine ; proche de 0 = équipartition.
        Population vide → 0.0. (Diversité de Simpson = 1 - λ.)
    """
    vals = [float(c) for c in counts if c > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return sum((v / total) ** 2 for v in vals)


def pianka_overlap(dist_a: Iterable[float], dist_b: Iterable[float]) -> float:
    """Recouvrement de niche de Pianka entre deux distributions d'utilisation.

    O_jk = Σ(p_ij p_ik) / sqrt(Σ p_ij² · Σ p_ik²), symétrique, ∈ [0, 1].

    Args:
        dist_a, dist_b: utilisation de chaque ressource/cellule (mêmes longueurs).
            Normalisées en interne ; tout vecteur non-négatif accepté.

    Returns:
        1.0 = niches identiques, 0.0 = niches disjointes. Si l'une des
        distributions est vide/nulle → 0.0 (pas de recouvrement défini).
    """
    a = [float(x) for x in dist_a]
    b = [float(x) for x in dist_b]
    if len(a) != len(b):
        raise ValueError(f"longueurs incompatibles: {len(a)} vs {len(b)}")
    sa, sb = sum(a), sum(b)
    if sa <= 0 or sb <= 0:
        return 0.0
    pa = [x / sa for x in a]
    pb = [x / sb for x in b]
    num = sum(x * y for x, y in zip(pa, pb))
    den = math.sqrt(sum(x * x for x in pa) * sum(y * y for y in pb))
    return num / den if den > 0 else 0.0


@dataclass
class BifurcationResult:
    """Résultat d'une détection naïve de point de rupture (changepoint)."""

    changed: bool
    index: int            # position du split (sur la série fournie), -1 si aucun
    score: float          # |moyenne_avant - moyenne_après| normalisé par l'écart-type global
    mean_before: float
    mean_after: float


def detect_bifurcation(
    series: list[float] | tuple[float, ...],
    *,
    min_segment: int = 3,
    threshold: float = 2.0,
) -> BifurcationResult:
    """Détecte un changement de niveau (shift de moyenne) dans une série 1D.

    Méthode : balaye tous les splits valides, choisit celui qui maximise
    l'écart de moyenne inter-segments, et le déclare significatif si cet
    écart dépasse ``threshold`` écarts-types globaux. Léger et déterministe,
    suffisant pour repérer extinctions/effondrements dans une courbe alive(t).

    Args:
        series: valeurs ordonnées (ex : population par snapshot).
        min_segment: taille minimale de chaque côté du split.
        threshold: nb d'écarts-types globaux requis pour déclarer un changement.

    Returns:
        BifurcationResult (``changed=False`` si série trop courte ou plate).
    """
    n = len(series)
    if n < 2 * min_segment:
        return BifurcationResult(False, -1, 0.0, 0.0, 0.0)
    xs = [float(v) for v in series]
    mean_all = sum(xs) / n
    var = sum((v - mean_all) ** 2 for v in xs) / n
    std = math.sqrt(var)
    if std == 0:
        return BifurcationResult(False, -1, 0.0, mean_all, mean_all)

    best_idx, best_gap, best_mb, best_ma = -1, 0.0, mean_all, mean_all
    for i in range(min_segment, n - min_segment + 1):
        left, right = xs[:i], xs[i:]
        mb = sum(left) / len(left)
        ma = sum(right) / len(right)
        gap = abs(mb - ma)
        if gap > best_gap:
            best_idx, best_gap, best_mb, best_ma = i, gap, mb, ma
    score = best_gap / std
    return BifurcationResult(
        changed=score >= threshold,
        index=best_idx,
        score=score,
        mean_before=best_mb,
        mean_after=best_ma,
    )


@dataclass
class EcologyTracker:
    """Accumule des observations spatiales par affinité et calcule les
    métriques d'écologie sur l'ensemble du run.

    Modèle : la « niche » d'une affinité est sa distribution d'occupation sur
    une grille grossière (super-cellules ``grid_bins × grid_bins``). Le
    recouvrement de niche entre deux affinités = Pianka sur ces distributions.

    Usage::

        tr = EcologyTracker(rows=64, cols=64, n_affinities=4)
        for ev in iter_events("events.jsonl"):
            tr.observe_event(ev)
        block = tr.finalize()   # dict prêt à embarquer dans un report
    """

    rows: int
    cols: int
    n_affinities: int = 4
    grid_bins: int = 8

    # occupation[aff] = dict {bin_index: count cumulé}
    _occ: dict[int, dict[int, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _alive_series: list[int] = field(default_factory=list)
    _lineage_series: list[int] = field(default_factory=list)
    _affinity_totals: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _n_obs: int = 0

    def _bin(self, r: int, c: int) -> int:
        br = min(self.grid_bins - 1, int(r * self.grid_bins / max(self.rows, 1)))
        bc = min(self.grid_bins - 1, int(c * self.grid_bins / max(self.cols, 1)))
        return br * self.grid_bins + bc

    def observe_agent(self, r: int, c: int, affinity: int) -> None:
        """Enregistre une position d'agent avec son affinité (0..n_affinities-1)."""
        self._occ[affinity][self._bin(r, c)] += 1
        self._affinity_totals[affinity] += 1
        self._n_obs += 1

    def observe_event(self, event: dict[str, Any]) -> None:
        """Consomme un event v8 (schema 2) : agents + n_alive + n_lin."""
        for a in event.get("agents", []):
            aff = a.get("aff", 0)
            if aff is None:
                aff = 0
            self.observe_agent(a["r"], a["c"], int(aff))
        if "n_alive" in event:
            self._alive_series.append(int(event["n_alive"]))
        if "n_lin" in event:
            self._lineage_series.append(int(event["n_lin"]))

    def niche_overlap_matrix(self) -> dict[str, float]:
        """Pianka entre chaque paire d'affinités peuplées. Clé ``"i-j"`` (i<j)."""
        n_cells = self.grid_bins * self.grid_bins
        dists = {}
        for aff in range(self.n_affinities):
            if self._occ.get(aff):
                vec = [0.0] * n_cells
                for b, cnt in self._occ[aff].items():
                    vec[b] = float(cnt)
                dists[aff] = vec
        out: dict[str, float] = {}
        affs = sorted(dists)
        for i in range(len(affs)):
            for j in range(i + 1, len(affs)):
                out[f"{affs[i]}-{affs[j]}"] = pianka_overlap(dists[affs[i]], dists[affs[j]])
        return out

    def finalize(self) -> dict[str, Any]:
        """Retourne le bloc de métriques d'écologie (dict sérialisable JSON)."""
        aff_counts = [self._affinity_totals.get(a, 0) for a in range(self.n_affinities)]
        overlaps = self.niche_overlap_matrix()
        mean_overlap = sum(overlaps.values()) / len(overlaps) if overlaps else 0.0
        bif = detect_bifurcation(self._alive_series) if self._alive_series else None
        return {
            "n_observations": self._n_obs,
            "affinity_counts": aff_counts,
            "shannon_diversity": shannon_diversity(aff_counts),
            "shannon_evenness": shannon_diversity(aff_counts, normalized=True),
            "simpson_dominance": simpson_dominance(aff_counts),
            "niche_overlap": overlaps,
            "mean_niche_overlap": mean_overlap,
            "mean_lineages": (
                sum(self._lineage_series) / len(self._lineage_series)
                if self._lineage_series else 0.0
            ),
            "max_lineages": max(self._lineage_series) if self._lineage_series else 0,
            "alive_bifurcation": (
                {
                    "changed": bif.changed,
                    "index": bif.index,
                    "score": round(bif.score, 4),
                    "mean_before": round(bif.mean_before, 2),
                    "mean_after": round(bif.mean_after, 2),
                }
                if bif is not None else None
            ),
        }
