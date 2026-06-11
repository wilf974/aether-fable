"""Agrégation multi-seeds des runs AetherLife (V2.5).

Scanne un dossier de runs (chacun contenant un report JSON overnight et/ou un
``run_summary.json``), extrait une métrique par chemin pointé, et agrège sur
l'ensemble des seeds. Pur stdlib.

Un "run" = un dossier contenant au moins un de :
- ``overnight_v8b1_seed*.json`` (report riche : final_state, ecology_v25, ...)
- ``run_summary.json`` (télémétrie MetricsLogger)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from aetherlife.analysis.stats import Summary, summarize

__all__ = ["load_run", "collect_runs", "get_path", "aggregate_metric"]


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Charge un run : fusionne run_summary.json puis le report overnight.

    Le report overnight (plus riche) écrase run_summary en cas de clé commune.
    Retourne {} si le dossier ne contient aucun fichier exploitable.
    """
    d = Path(run_dir)
    merged: dict[str, Any] = {}
    summ = d / "run_summary.json"
    if summ.exists():
        merged.update(_read_json(summ))
    reports = sorted(d.glob("overnight_v8b1_seed*.json"))
    for r in reports:
        merged.update(_read_json(r))
    # report générique fallback (tout autre *.json à la racine sauf config/summary/metrics)
    if not reports:
        for r in sorted(d.glob("*.json")):
            if r.name not in ("run_summary.json", "run_config.json", "metrics.jsonl"):
                merged.update(_read_json(r))
    return merged


def collect_runs(root: str | Path) -> list[dict[str, Any]]:
    """Trouve récursivement les dossiers de runs sous ``root`` et les charge.

    Un dossier est un run s'il contient run_summary.json ou un report *.json.
    """
    root = Path(root)
    seen: set[Path] = set()
    runs: list[dict[str, Any]] = []
    patterns = ("run_summary.json", "overnight_v8b1_seed*.json")
    for pat in patterns:
        for f in root.rglob(pat):
            if f.parent in seen:
                continue
            seen.add(f.parent)
            data = load_run(f.parent)
            if data:
                runs.append(data)
    return runs


def get_path(data: dict, path: str, default: Any = None) -> Any:
    """Lit une valeur via un chemin pointé, ex ``"ecology_v25.shannon_diversity"``.

    Supporte les index de liste : ``"top_lineages.0.pct"``.
    """
    cur: Any = data
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and key.lstrip("-").isdigit():
            i = int(key)
            if -len(cur) <= i < len(cur):
                cur = cur[i]
            else:
                return default
        else:
            return default
    return cur


def aggregate_metric(
    runs: Iterable[dict[str, Any]],
    path: str,
    confidence: float = 0.95,
) -> Summary:
    """Extrait ``path`` (numérique) de chaque run et résume l'échantillon.

    Les runs où la valeur est absente ou non numérique sont ignorés.
    """
    vals: list[float] = []
    for r in runs:
        v = get_path(r, path)
        if isinstance(v, bool):
            vals.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            vals.append(float(v))
    return summarize(vals, confidence=confidence)
