"""Analyse mécaniste de l'effet protecteur diversité→survie (C2).

Teste H1 (diversification spatiale) vs H3 (sauvetage démographique par réservoirs
asynchrones) sur les trajectoires fines k=1 vs k=4 (record_events_v8 ticks~4000,
record_every 5), à travers le goulot démographique.

Question : k=4 survit-il parce qu'il maintient PLUSIEURS réservoirs (affinité×biome)
ASYNCHRONES qui amortissent le crash, là où k=1 n'a qu'un réservoir unique qui
s'effondre en bloc ?

Usage:
    python scripts/analyze_c2_mechanism.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys

sys.path.insert(0, "scripts")
from overnight_v8b1 import build_env  # noqa: E402


def _load(d: str):
    with open(f"{d}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    evs = []
    with open(f"{d}/events.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                evs.append(json.loads(line))
    return meta, evs


_BIOME_CACHE: dict[int, object] = {}


def _bmap(seed: int):
    if seed not in _BIOME_CACHE:
        e = build_env(seed, regime="coordination_collective",
                      vocalize_energy_cost=0.05)
        if e.biome_map.sum() == 0:
            e.reset(seed=seed)
        _BIOME_CACHE[seed] = e.biome_map
    return _BIOME_CACHE[seed]


def analyze_run(d: str) -> dict:
    meta, evs = _load(d)
    seed = meta["seed"]
    k = meta.get("n_initial_affinities")
    alive = [(e["t"], e["n_alive"]) for e in evs]
    extinct = alive[-1][1] == 0 if alive else True
    min_alive = min(n for _, n in alive)
    t_min = next(t for t, n in alive if n == min_alive)
    max_after = max((n for t, n in alive if t > t_min), default=0)

    # Réservoirs au creux : nb d'affinités vivantes, nb de (biome occupé) distincts
    creux_frame = min(evs, key=lambda e: e["n_alive"])
    affs_at_creux = len(set(a["aff"] for a in creux_frame["agents"]))
    bm = _bmap(seed)
    biomes_at_creux = len(set(int(bm[a["r"], a["c"]]) for a in creux_frame["agents"]))

    # Synchronie d'extinction : les affinités s'effondrent-elles ENSEMBLE (k=1, un
    # seul réservoir) ou en décalé (k=4) ? On mesure l'écart-type des temps où
    # chaque affinité atteint son minimum de population.
    aff_min_ticks: dict[int, int] = {}
    aff_series: dict[int, list[tuple[int, int]]] = {}
    for e in evs:
        cnt: dict[int, int] = {}
        for a in e["agents"]:
            cnt[a["aff"]] = cnt.get(a["aff"], 0) + 1
        for aff, c in cnt.items():
            aff_series.setdefault(aff, []).append((e["t"], c))
    for aff, series in aff_series.items():
        mn = min(c for _, c in series)
        aff_min_ticks[aff] = next(t for t, c in series if c == mn)
    crash_async = (
        st.pstdev(list(aff_min_ticks.values()))
        if len(aff_min_ticks) >= 2 else 0.0
    )

    # Pente de recovery : (max_after - min) / (t où max_after atteint - t_min)
    recovery_slope = None
    if max_after > min_alive:
        t_rec = next(t for t, n in alive if t > t_min and n == max_after)
        if t_rec > t_min:
            recovery_slope = (max_after - min_alive) / (t_rec - t_min)

    return {
        "seed": seed, "k": k, "extinct": extinct,
        "min_alive": min_alive, "t_min": t_min,
        "affs_at_creux": affs_at_creux, "biomes_at_creux": biomes_at_creux,
        "crash_async": round(crash_async, 1),
        "recovery_slope": round(recovery_slope, 3) if recovery_slope else None,
    }


def main() -> None:
    rows = []
    for k in (1, 4):
        for d in sorted(glob.glob(f"results/c2dyn_aff{k}/seed*")):
            try:
                rows.append(analyze_run(d))
            except Exception as ex:  # noqa: BLE001
                print(f"  skip {d}: {ex}")
    print(f"{'seed':>4} {'k':>2} {'ext':>4} {'minAl':>6} {'tMin':>5} "
          f"{'affCrx':>7} {'biomCrx':>8} {'async':>6} {'recov':>6}")
    for r in rows:
        print(f"{r['seed']:>4} {r['k']:>2} {str(r['extinct'])[0]:>4} "
              f"{r['min_alive']:>6} {r['t_min']:>5} {r['affs_at_creux']:>7} "
              f"{r['biomes_at_creux']:>8} {r['crash_async']:>6} "
              f"{str(r['recovery_slope']):>6}")
    print()
    for k in (1, 4):
        sub = [r for r in rows if r["k"] == k]
        if not sub:
            continue
        ext = sum(r["extinct"] for r in sub)
        print(f"k={k} (n={len(sub)}): extinction={ext}/{len(sub)}  "
              f"min_alive_moy={st.mean(r['min_alive'] for r in sub):.1f}  "
              f"affs_creux_moy={st.mean(r['affs_at_creux'] for r in sub):.1f}  "
              f"biomes_creux_moy={st.mean(r['biomes_at_creux'] for r in sub):.1f}  "
              f"crash_async_moy={st.mean(r['crash_async'] for r in sub):.0f}")


if __name__ == "__main__":
    main()
