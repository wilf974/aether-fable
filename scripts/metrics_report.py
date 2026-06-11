"""metrics_report — résumé des runs télémétrie (metrics.jsonl).

Usage :
    python scripts/metrics_report.py results/run_seed7
    python scripts/metrics_report.py results/c2_aff*          # multi-runs
    python scripts/metrics_report.py results/run_seed7 --plot # PNG (matplotlib requis)

Lit metrics.jsonl (+ run_summary.json / run_config.json si présents) et
affiche : durée, points de mesure, dernière valeur / min / max de chaque
métrique numérique. Avec --plot, trace chaque métrique vs step dans
<run_dir>/metrics.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_run(run_dir: Path) -> dict:
    """Charge un run : {'records': [...], 'summary': {...}, 'config': {...}}."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"pas de metrics.jsonl dans {run_dir}")
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {"dir": run_dir, "records": records, "summary": {}, "config": {}}
    for name in ("run_summary", "run_config"):
        p = run_dir / f"{name}.json"
        if p.exists():
            out[name.split("_")[1]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def numeric_series(records: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Extrait {metrique: [(step, valeur), ...]} pour les valeurs numériques."""
    skip = {"run_id", "step", "wall_time", "phase"}
    series: dict[str, list[tuple[int, float]]] = {}
    for r in records:
        for k, v in r.items():
            if k in skip or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            series.setdefault(k, []).append((r["step"], float(v)))
    return series


def print_report(run: dict) -> None:
    records = run["records"]
    print(f"\n=== {run['dir']} ===")
    if not records:
        print("  (vide)")
        return
    run_id = records[0].get("run_id", "?")
    wall = records[-1].get("wall_time", 0.0)
    steps = [r["step"] for r in records]
    print(f"  run_id={run_id}  points={len(records)}  "
          f"steps {min(steps)}..{max(steps)}  wall={wall:.0f}s")
    if run["config"]:
        keys = ("seed", "regime", "n_ticks", "device")
        cfg = "  ".join(f"{k}={run['config'][k]}" for k in keys if k in run["config"])
        if cfg:
            print(f"  config : {cfg}")
    series = numeric_series(records)
    if series:
        w = max(len(k) for k in series)
        print(f"  {'métrique'.ljust(w)}   dernière       min       max    n")
        for k, pts in sorted(series.items()):
            vals = [v for _, v in pts]
            print(f"  {k.ljust(w)}  {vals[-1]:9.4g} {min(vals):9.4g} "
                  f"{max(vals):9.4g}  {len(vals):3d}")
    if run["summary"]:
        interesting = {k: v for k, v in run["summary"].items() if k != "run_id"}
        print(f"  summary : {json.dumps(interesting, ensure_ascii=False, default=str)}")


def plot_run(run: dict) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  --plot ignoré : matplotlib non installé", file=sys.stderr)
        return None
    series = numeric_series(run["records"])
    if not series:
        return None
    n = len(series)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    for ax, (name, pts) in zip(axes.flat, sorted(series.items())):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, lw=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    run_id = run["records"][0].get("run_id", "run")
    fig.suptitle(f"AetherLife — {run_id}")
    fig.tight_layout()
    out = run["dir"] / "metrics.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  plot : {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="dossier(s) de run contenant metrics.jsonl")
    ap.add_argument("--plot", action="store_true", help="génère metrics.png par run")
    args = ap.parse_args(argv)

    n_ok = 0
    for d in args.run_dirs:
        try:
            run = load_run(d)
        except FileNotFoundError as e:
            print(f"skip {d} : {e}", file=sys.stderr)
            continue
        print_report(run)
        if args.plot:
            plot_run(run)
        n_ok += 1
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
