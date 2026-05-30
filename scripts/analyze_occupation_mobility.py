"""Mesure village-vs-migration sur un (ou plusieurs) clip events.jsonl.

Critère (défini avec l'utilisateur 2026-05-30) :
    corr occupation début~fin > 0.8  → VILLAGE (agrégation sédentaire fixe)
    corr occupation début~fin < 0.5  → MIGRATION (agrégation qui se relocalise)
    entre les deux                   → dérive partielle

Compagnon du V8 Replay Viewer : exploite l'events.jsonl produit par
record_events_v8.py (positions par tick que les rapports agrégés n'ont pas).

Usage:
    python scripts/analyze_occupation_mobility.py results/clip_seed25 results/clip_seed14 ...
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys

BINS = 8  # grille 8x8 de super-cellules


def _load(clip_dir: str) -> tuple[dict, list[dict]]:
    with open(f"{clip_dir}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    evs = []
    with open(f"{clip_dir}/events.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                evs.append(json.loads(line))
    return meta, evs


def _occ_hist(frames: list[dict], rows: int, cols: int) -> list[float]:
    bs_r = max(1, rows // BINS)
    bs_c = max(1, cols // BINS)
    h = [0.0] * (BINS * BINS)
    n = 0
    for e in frames:
        for a in e["agents"]:
            bi = min(BINS - 1, a["r"] // bs_r) * BINS + min(BINS - 1, a["c"] // bs_c)
            h[bi] += 1
            n += 1
    return [x / n for x in h] if n else h


def _corr(x: list[float], y: list[float]) -> float:
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx * dy else 0.0


def _verdict(corr_first_last: float) -> str:
    if corr_first_last > 0.8:
        return "VILLAGE (sedentaire)"
    if corr_first_last < 0.5:
        return "MIGRATION (relocalisation)"
    return "derive partielle"


def analyze(clip_dir: str) -> dict:
    meta, evs = _load(clip_dir)
    rows, cols = int(meta["rows"]), int(meta["cols"])
    n = len(evs)
    if n < 3:
        return {"clip": clip_dir, "error": "trop peu de frames"}
    third = n // 3
    early, late = evs[:third], evs[-third:]
    h_e = _occ_hist(early, rows, cols)
    h_l = _occ_hist(late, rows, cols)
    corr = _corr(h_e, h_l)

    coms = []
    for e in evs:
        if not e["agents"]:
            continue
        cr = st.mean(a["r"] for a in e["agents"])
        cc = st.mean(a["c"] for a in e["agents"])
        coms.append((cr, cc))
    path = sum(math.dist(coms[i], coms[i - 1]) for i in range(1, len(coms)))
    net = math.dist(coms[0], coms[-1]) if coms else 0.0

    # Creux démographique (driver candidat de la migration)
    alive = [e["n_alive"] for e in evs if "n_alive" in e]
    min_alive = min(alive) if alive else 0
    min_tick = next((e["t"] for e in evs if e.get("n_alive") == min_alive), None)

    return {
        "clip": clip_dir,
        "seed": meta.get("seed"),
        "frames": n,
        "corr_early_late": round(corr, 3),
        "min_alive": min_alive,
        "min_tick": min_tick,
        "com_net": round(net, 1),
        "verdict": _verdict(corr),
    }


def main() -> None:
    clips = sys.argv[1:]
    if not clips:
        print("usage: analyze_occupation_mobility.py <clip_dir> [clip_dir ...]")
        sys.exit(1)
    print(f"{'seed':>5} {'frames':>6} {'corr_e~l':>9} {'minAlive':>8} "
          f"{'minTick':>8} {'com_net':>8}  verdict")
    rows = []
    for c in clips:
        r = analyze(c)
        if "error" in r:
            print(f"  {c}: {r['error']}")
            continue
        rows.append(r)
        print(f"{str(r['seed']):>5} {r['frames']:>6} {r['corr_early_late']:>9} "
              f"{r['min_alive']:>8} {str(r['min_tick']):>8} {r['com_net']:>8}  "
              f"{r['verdict']}")

    # Synthèse : taux des modes + test du driver creux↔migration
    if len(rows) >= 2:
        from collections import Counter
        modes = Counter(r["verdict"].split()[0] for r in rows)
        print(f"\n--- {len(rows)} seeds : "
              + "  ".join(f"{k}={v}" for k, v in modes.items()) + " ---")
        mig = [r["min_alive"] for r in rows if r["verdict"].startswith("MIGRATION")]
        vil = [r["min_alive"] for r in rows if r["verdict"].startswith("VILLAGE")]
        if mig and vil:
            print(f"creux moyen - MIGRATION={st.mean(mig):.1f}  "
                  f"VILLAGE={st.mean(vil):.1f}  "
                  f"(driver: creux profond -> migration ?)")


if __name__ == "__main__":
    main()
