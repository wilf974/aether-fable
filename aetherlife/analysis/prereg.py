"""Préenregistrement auditable d'expériences AetherLife (V2.5).

Inspiré du pattern « contrat auditable » d'AetherMind_OS (MissionProfile →
ExperimentPlan gaté + JSON auditable), adapté aux préenregistrements
multi-seeds d'AetherLife (cf. docs/preregistrations/).

Idée : figer AVANT collecte une hypothèse, ses conditions, ses seeds, sa
métrique primaire et ses critères de décision dans un spec JSON verrouillé.
Après collecte, ``audit()`` confronte les résultats agrégés aux critères
PRÉ-spécifiés et rend un verdict reproductible — jamais de p-hacking possible
puisque les seuils sont dans le spec figé.

Pur stdlib.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from aetherlife.analysis.aggregate import aggregate_metric
from aetherlife.analysis.stats import Summary

__all__ = [
    "Comparator", "Criterion", "Condition", "PreregSpec",
    "Verdict", "AuditResult", "audit",
]


class Comparator(str, Enum):
    """Opérateur d'un critère de décision pré-spécifié."""

    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="

    def test(self, value: float, threshold: float) -> bool:
        if self is Comparator.GT:
            return value > threshold
        if self is Comparator.GE:
            return value >= threshold
        if self is Comparator.LT:
            return value < threshold
        return value <= threshold


@dataclass(frozen=True)
class Condition:
    """Une condition expérimentale = un label + un override de config du runner.

    ``overrides`` est passé tel quel au lanceur (ex {"n_initial_affinities": 1}).
    """

    label: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label de condition vide")


@dataclass(frozen=True)
class Criterion:
    """Critère de décision pré-spécifié sur une métrique agrégée.

    Ex : metric="extinction_rate", comparator=LT, threshold=0.2 sur la
    condition "k4" ⇒ « le taux d'extinction k=4 doit être < 20 % ».
    """

    name: str
    metric_path: str
    comparator: Comparator
    threshold: float
    condition_label: str | None = None   # None = métrique agrégée tous runs confondus

    def __post_init__(self) -> None:
        if not self.metric_path:
            raise ValueError("metric_path vide")


@dataclass(frozen=True)
class PreregSpec:
    """Spec de préenregistrement figé. Sérialisable JSON, round-trippable."""

    prereg_id: str
    hypothesis: str
    primary_metric: str
    conditions: tuple[Condition, ...]
    criteria: tuple[Criterion, ...]
    seeds: tuple[int, ...]
    regime: str = "coordination_collective"
    n_ticks: int = 16_000
    device: str = "cuda"
    locked_at: str | None = None          # ISO8601 ; None = pas encore verrouillé
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.prereg_id.strip():
            raise ValueError("prereg_id vide")
        if not self.conditions:
            raise ValueError("au moins une condition requise")
        if not self.seeds:
            raise ValueError("au moins un seed requis")
        labels = [c.label for c in self.conditions]
        if len(labels) != len(set(labels)):
            raise ValueError("labels de conditions non uniques")
        for cr in self.criteria:
            if cr.condition_label is not None and cr.condition_label not in labels:
                raise ValueError(
                    f"critère {cr.name!r} référence une condition inconnue "
                    f"{cr.condition_label!r}"
                )

    # ── verrouillage / sérialisation ───────────────────────────────────

    def locked(self, when: str | None = None) -> "PreregSpec":
        """Retourne une copie verrouillée (locked_at renseigné)."""
        stamp = when or time.strftime("%Y-%m-%dT%H:%M:%S")
        d = self.to_dict()
        d["locked_at"] = stamp
        return PreregSpec.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prereg_id": self.prereg_id,
            "hypothesis": self.hypothesis,
            "primary_metric": self.primary_metric,
            "conditions": [{"label": c.label, "overrides": c.overrides}
                           for c in self.conditions],
            "criteria": [{"name": cr.name, "metric_path": cr.metric_path,
                          "comparator": cr.comparator.value, "threshold": cr.threshold,
                          "condition_label": cr.condition_label}
                         for cr in self.criteria],
            "seeds": list(self.seeds),
            "regime": self.regime,
            "n_ticks": self.n_ticks,
            "device": self.device,
            "locked_at": self.locked_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreregSpec":
        return cls(
            prereg_id=d["prereg_id"],
            hypothesis=d.get("hypothesis", ""),
            primary_metric=d["primary_metric"],
            conditions=tuple(Condition(c["label"], dict(c.get("overrides", {})))
                             for c in d["conditions"]),
            criteria=tuple(Criterion(cr["name"], cr["metric_path"],
                                     Comparator(cr["comparator"]), float(cr["threshold"]),
                                     cr.get("condition_label"))
                           for cr in d.get("criteria", [])),
            seeds=tuple(int(s) for s in d["seeds"]),
            regime=d.get("regime", "coordination_collective"),
            n_ticks=int(d.get("n_ticks", 16_000)),
            device=d.get("device", "cuda"),
            locked_at=d.get("locked_at"),
            notes=d.get("notes", ""),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PreregSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ── plan de lancement reproductible ────────────────────────────────

    def launch_plan(self, out_root: str = "results") -> list[dict[str, Any]]:
        """Génère la liste reproductible des runs (condition × seed) à lancer.

        Chaque entrée : label, seed, out_dir, et la commande overnight complète.
        Aucune exécution — uniquement le plan auditable.
        """
        plan: list[dict[str, Any]] = []
        for cond in self.conditions:
            for seed in self.seeds:
                out_dir = f"{out_root}/{self.prereg_id}/{cond.label}/seed{seed}"
                cmd = [
                    "python", "scripts/overnight_v8b1.py",
                    "--seed", str(seed), "--ticks", str(self.n_ticks),
                    "--regime", self.regime, "--device", self.device,
                    "--out-dir", out_dir,
                ]
                for k, v in cond.overrides.items():
                    cmd += [f"--{k.replace('_', '-')}", str(v)]
                plan.append({"label": cond.label, "seed": seed,
                             "out_dir": out_dir, "command": cmd})
        return plan


class Verdict(str, Enum):
    SUCCESS = "SUCCÈS"
    PARTIAL = "PARTIEL"
    FAILURE = "ÉCHEC"
    INCOMPLETE = "INCOMPLET"   # données manquantes


@dataclass(frozen=True)
class CriterionResult:
    name: str
    metric_path: str
    condition_label: str | None
    expected: str               # ex ">= 0.30"
    observed: float | None
    n: int
    passed: bool


@dataclass(frozen=True)
class AuditResult:
    verdict: Verdict
    criteria: tuple[CriterionResult, ...]
    n_pass: int
    n_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "n_pass": self.n_pass,
            "n_total": self.n_total,
            "criteria": [
                {"name": c.name, "metric_path": c.metric_path,
                 "condition": c.condition_label, "expected": c.expected,
                 "observed": c.observed, "n": c.n, "passed": c.passed}
                for c in self.criteria
            ],
        }


def audit(
    spec: PreregSpec,
    runs_by_condition: dict[str, list[dict[str, Any]]],
    confidence: float = 0.95,
) -> AuditResult:
    """Confronte les runs agrégés aux critères PRÉ-spécifiés du spec.

    Args:
        spec: le préenregistrement figé.
        runs_by_condition: {label_condition: [run_dict, ...]}.
        confidence: niveau pour les IC des métriques agrégées.

    Returns:
        AuditResult avec le verdict global (SUCCESS = tous critères passés ;
        PARTIAL = au moins un mais pas tous ; FAILURE = aucun ;
        INCOMPLETE = au moins un critère sans données).
    """
    results: list[CriterionResult] = []
    for cr in spec.criteria:
        if cr.condition_label is not None:
            runs = runs_by_condition.get(cr.condition_label, [])
        else:
            runs = [r for rs in runs_by_condition.values() for r in rs]
        summ: Summary = aggregate_metric(runs, cr.metric_path, confidence)
        observed = summ.mean if summ.n > 0 else None
        passed = observed is not None and cr.comparator.test(observed, cr.threshold)
        results.append(CriterionResult(
            name=cr.name, metric_path=cr.metric_path,
            condition_label=cr.condition_label,
            expected=f"{cr.comparator.value} {cr.threshold}",
            observed=observed, n=summ.n, passed=passed,
        ))
    n_total = len(results)
    n_pass = sum(1 for r in results if r.passed)
    has_missing = any(r.observed is None for r in results)
    if has_missing:
        verdict = Verdict.INCOMPLETE
    elif n_pass == n_total:
        verdict = Verdict.SUCCESS
    elif n_pass == 0:
        verdict = Verdict.FAILURE
    else:
        verdict = Verdict.PARTIAL
    return AuditResult(verdict, tuple(results), n_pass, n_total)

