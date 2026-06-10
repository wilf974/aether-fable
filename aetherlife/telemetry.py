"""Télémétrie structurée AetherLife (V2.5).

Deux briques, zéro dépendance externe :

- ``get_logger`` : logger stdlib configuré (console, fichier optionnel),
  idempotent — appeler plusieurs fois ne duplique pas les handlers.
- ``MetricsLogger`` : série temporelle JSONL append-only, flush à chaque
  écriture (crash-safe : un run overnight tué laisse un fichier lisible).

Format JSONL : un objet par ligne, clés minimales ``run_id``, ``step``,
``wall_time`` (secondes depuis création du logger) + métriques libres.
Lecture : ``[json.loads(l) for l in open("metrics.jsonl")]`` ou
``pandas.read_json(path, lines=True)``.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str = "aetherlife",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Logger console (stdout) + fichier optionnel. Idempotent."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    has_console = any(
        getattr(h, "_aetherlife_console", False) for h in logger.handlers
    )
    if not has_console:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        h._aetherlife_console = True  # type: ignore[attr-defined]
        logger.addHandler(h)

    if log_file is not None:
        log_path = str(Path(log_file).resolve())
        has_file = any(
            getattr(h, "baseFilename", None) == log_path for h in logger.handlers
        )
        if not has_file:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
            logger.addHandler(fh)

    return logger


def _jsonable(obj: Any) -> Any:
    """Fallback de sérialisation : numpy scalars/arrays, Path, sets, etc."""
    if hasattr(obj, "tolist"):  # numpy array ou scalar
        return obj.tolist()
    if hasattr(obj, "item"):  # autres scalars numpy-like
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


class MetricsLogger:
    """Écrit ``metrics.jsonl`` (+ ``run_config.json``, ``run_summary.json``)
    dans ``out_dir``. Append-only, flush à chaque ``log()``.

    Usage::

        with MetricsLogger(out_dir, run_id="c2_seed7", config=vars(args)) as ml:
            for t in range(n_ticks):
                ...
                if t % snap_every == 0:
                    ml.log(t, alive=env.n_alive, mean_loss=loss)
            ml.summary(duration_s=dt, final_alive=env.n_alive)
    """

    def __init__(
        self,
        out_dir: str | Path,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.path = self.out_dir / "metrics.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._t0 = time.time()
        self._closed = False
        if config:
            (self.out_dir / "run_config.json").write_text(
                json.dumps(config, indent=2, default=_jsonable, ensure_ascii=False),
                encoding="utf-8",
            )

    def log(self, step: int, **metrics: Any) -> None:
        """Ajoute un point de mesure. Flush immédiat (crash-safe)."""
        if self._closed:
            raise RuntimeError("MetricsLogger fermé")
        rec = {
            "run_id": self.run_id,
            "step": int(step),
            "wall_time": round(time.time() - self._t0, 3),
            **metrics,
        }
        self._fh.write(json.dumps(rec, default=_jsonable, ensure_ascii=False) + "\n")
        self._fh.flush()

    def summary(self, **fields: Any) -> Path:
        """Écrit ``run_summary.json`` (état final du run)."""
        path = self.out_dir / "run_summary.json"
        payload = {
            "run_id": self.run_id,
            "duration_s": round(time.time() - self._t0, 3),
            **fields,
        }
        path.write_text(
            json.dumps(payload, indent=2, default=_jsonable, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
