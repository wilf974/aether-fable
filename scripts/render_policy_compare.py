"""OBS V3.0 — Policy Distance + heatmaps comparatives village vs mobile.

Charge N JSON (probe_policies_v8), sépare village/mobile, calcule la distance
intra-groupe vs inter-groupe (test H2 vs H3), rend des heatmaps PNG.

Usage:
    python scripts/render_policy_compare.py results/probe/*.json \
        --out clips/policy_compare.png
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aetherlife.viz.policy_probe import policy_distance  # noqa: E402

_CELL = 28
_PAD = 4


def _heat_color(v: float, lo: float, hi: float):
    t = 0.0 if hi == lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    return (int(30 + 200 * t), int(30 + 120 * t), int(80 + 60 * (1 - t)))


def _draw_fp(surf, fp, x0, y0, lo, hi):
    for i, row in enumerate(fp):
        for j, v in enumerate(row):
            rect = pygame.Rect(x0 + j * _CELL, y0 + i * _CELL,
                               _CELL - 1, _CELL - 1)
            pygame.draw.rect(surf, _heat_color(v, lo, hi), rect)


def compare(paths: list[str], out_png: str) -> dict:
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs.append(json.load(f))
    vil = [np.array(r["fingerprint"]) for r in recs if r["village_basin"]]
    mob = [np.array(r["fingerprint"]) for r in recs if not r["village_basin"]]

    def _pairs(group):
        return [policy_distance(group[i], group[j])
                for i in range(len(group)) for j in range(i + 1, len(group))]
    intra = _pairs(vil) + _pairs(mob)
    inter = [policy_distance(a, b) for a in vil for b in mob]
    intra_m = st.mean(intra) if intra else 0.0
    inter_m = st.mean(inter) if inter else 0.0

    pygame.init()
    n_p = len(recs[0]["probe_labels"]) if recs else 0
    w = 2 * (9 * _CELL) + 3 * _PAD + 40
    h = n_p * _CELL + 60
    surf = pygame.Surface((w, h))
    surf.fill((18, 18, 20))
    if vil:
        fv = np.mean(vil, axis=0)
        lo, hi = float(fv.min()), float(fv.max())
        _draw_fp(surf, fv, _PAD, 40, lo, hi)
    if mob:
        fm = np.mean(mob, axis=0)
        lo, hi = float(fm.min()), float(fm.max())
        _draw_fp(surf, fm, 9 * _CELL + 2 * _PAD, 40, lo, hi)
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 14)
    surf.blit(font.render(
        f"VILLAGE (n={len(vil)})   |   MOBILE (n={len(mob)})", True,
        (220, 220, 225)), (_PAD, 6))
    surf.blit(font.render(
        f"intra={intra_m:.3f}  inter={inter_m:.3f}  "
        f"verdict={'H2' if inter_m > 1.3 * intra_m else 'H3'}",
        True, (220, 220, 225)), (_PAD, 22))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    pygame.image.save(surf, out_png)

    return {
        "n_village": len(vil), "n_mobile": len(mob),
        "intra": round(intra_m, 4), "inter": round(inter_m, 4),
        "verdict": "H2" if inter_m > 1.3 * intra_m else "H3",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="clips/policy_compare.png")
    a = p.parse_args()
    res = compare(a.paths, a.out)
    print(f"village={res['n_village']} mobile={res['n_mobile']}")
    print(f"intra-group distance = {res['intra']}")
    print(f"inter-group distance = {res['inter']}")
    print(f"VERDICT : {res['verdict']}  (inter >> intra -> H2 ; "
          f"inter ~ intra -> H3)")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
