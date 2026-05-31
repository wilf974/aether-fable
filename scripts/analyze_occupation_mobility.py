"""Mesure officielle de mobilité spatiale sur des clips events.jsonl.

SOURCE DE VÉRITÉ UNIQUE : `aetherlife.historian.spatial_mobility` — exactement
la même métrique que l'Historien (fenêtres 10 % début/fin, occupation 8×8,
corr de Pearson, seuil bassin village 0.8). La définition n'est PAS dupliquée
ici : on importe le helper.

`mobility_score = corr_occupation_start_end` (continu ; ~1 = village sédentaire,
plus bas = mobilité collective). Cf. finding 2026-05-30 coordination-mobility-modes.

Nuance : le recorder échantillonne les positions (record_every) ; overnight les
voit à chaque tick. La métrique (définition) est identique ; seule la densité
d'échantillonnage diffère — sans impact sur la corr d'une distribution.

Usage:
    python scripts/analyze_occupation_mobility.py results/clip_seed25 [clip_dir ...]
"""
from __future__ import annotations

import json
import statistics as st
import sys

from aetherlife.historian.spatial_mobility import (
    OccupancyAccumulator,
    build_spatial_mobility_block,
    window_bounds,
)


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


def analyze(clip_dir: str) -> dict:
    meta, evs = _load(clip_dir)
    if not evs:
        return {"clip": clip_dir, "error": "events.jsonl vide"}
    rows, cols = int(meta["rows"]), int(meta["cols"])
    total = int(meta.get("total_ticks") or evs[-1]["t"])
    (s0, s1), (e0, e1) = window_bounds(total)  # tiers — identique à l'Historien

    start = OccupancyAccumulator(rows, cols)
    end = OccupancyAccumulator(rows, cols)
    for e in evs:
        t = e["t"]
        if s0 < t <= s1:
            start.add_positions([(a["r"], a["c"]) for a in e["agents"]])
        elif e0 < t <= e1:
            end.add_positions([(a["r"], a["c"]) for a in e["agents"]])

    block = build_spatial_mobility_block(
        start, end, start_window=(s0, s1), end_window=(e0, e1),
    )
    alive = [e["n_alive"] for e in evs if "n_alive" in e]
    block["seed"] = meta.get("seed")
    block["min_alive"] = min(alive) if alive else None
    return block


def main() -> None:
    clips = sys.argv[1:]
    if not clips:
        print("usage: analyze_occupation_mobility.py <clip_dir> [clip_dir ...]")
        sys.exit(1)

    print(f"{'seed':>5} {'mobility_score':>14} {'village':>8} {'creux':>6}  notes")
    rows = []
    for c in clips:
        r = analyze(c)
        if "error" in r:
            print(f"  {c}: {r['error']}")
            continue
        rows.append(r)
        score = r["corr_occupation_start_end"]
        score_s = f"{score:.3f}" if score is not None else "None"
        note = (
            "extinction" if score is None
            else "village" if r["village_basin"]
            else "mobile"
        )
        print(f"{str(r['seed']):>5} {score_s:>14} {str(r['village_basin']):>8} "
              f"{str(r['min_alive']):>6}  {note}")

    scored = [r for r in rows if r["corr_occupation_start_end"] is not None]
    if len(scored) >= 2:
        scores = [r["corr_occupation_start_end"] for r in scored]
        n_village = sum(1 for r in scored if r["village_basin"])
        print(f"\n--- {len(scored)} seeds notes : "
              f"village_basin={n_village} ({100*n_village/len(scored):.0f}%)  "
              f"hors_bassin={len(scored)-n_village} ---")
        print(f"mobility_score : mean={st.mean(scores):.3f}  "
              f"median={st.median(scores):.3f}  "
              f"min={min(scores):.3f}  max={max(scores):.3f}")


if __name__ == "__main__":
    main()
