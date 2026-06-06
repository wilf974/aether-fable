# Portfolio Generality — Spatial Topology — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tester si l'effet portefeuille (diversité→survie) survit à différentes granularités spatiales du monde, via un lever `n_seed_points ∈ {4,8,16}` croisé avec l'intervention de diversité `k ∈ {1,4}`.

**Architecture:** Jumeau de C2. Un lever additif `n_seed_points` exposé dans `build_env` (le champ existe déjà dans `BiomeConfig`, il est juste figé à 8 dans le régime coordination). Un runner grille (k×n) et un agrégateur qui sort la grille d'extinction et tranche H_spatial vs H_type.

**Tech Stack:** Python 3.13, numpy, pytest. Runs overnight CUDA (batch séparé).

**Spec :** `docs/superpowers/specs/2026-06-06-portfolio-generality-spatial-topology-design.md`

**Faits API (vérifiés)** :
- `BiomeConfig` a DÉJÀ le champ `n_seed_points: int` (défaut 8) — pas de nouveau champ à créer. Dans `build_env`, le `BiomeConfig` du régime coordination le fige à `n_seed_points=8` (ligne ~200).
- `n_initial_affinities` (C2) sert de patron exact : exposé dans `build_env` (sig ~ligne 65), passé au BiomeConfig (~ligne 218), `run_overnight` (sig ~389, appel ~397), report config (~677), CLI main (~940).
- `env.cfg.biomes.n_seed_points` lit la valeur. `env.biome_map` (généré au reset, déterministe par (seed, n_seed_points)).
- `extract_c2(report)` dans `scripts/aggregate_c2.py` lit `config.n_initial_affinities` + extinction — patron pour `aggregate_topology`.

---

## File Structure

| Fichier | Responsabilité |
|---|---|
| `scripts/overnight_v8b1.py` | exposer `n_seed_points` (build_env/run_overnight/CLI/report) — additif |
| `scripts/run_portfolio_topology.ps1` | runner grille k{1,4} × n{4,8,16} × seeds, idempotent |
| `scripts/aggregate_topology.py` | grille extinction par (k, n_seed_points) + verdict H_spatial/H_type |
| `tests/test_topology_generality.py` | propagation + reset + agrégation (sans GPU) |

---

## Task 1: Lever `n_seed_points` (build_env / run_overnight / CLI)

**Files:**
- Modify: `scripts/overnight_v8b1.py`
- Test: `tests/test_topology_generality.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_topology_generality.py` :

```python
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import numpy as np


def test_build_env_default_n_seed_points_is_8():
    from overnight_v8b1 import build_env
    env = build_env(1, regime="coordination_collective")
    assert env.cfg.biomes.n_seed_points == 8


def test_build_env_propagates_n_seed_points():
    from overnight_v8b1 import build_env
    env = build_env(1, regime="coordination_collective", n_seed_points=16)
    assert env.cfg.biomes.n_seed_points == 16


def _boundaries(bm):
    # nb de paires de cellules adjacentes de biomes différents (fragmentation)
    h = int((bm[:, :-1] != bm[:, 1:]).sum())
    v = int((bm[:-1, :] != bm[1:, :]).sum())
    return h + v


def test_higher_n_seed_points_more_fragmented():
    from overnight_v8b1 import build_env
    e4 = build_env(1, regime="coordination_collective", n_seed_points=4)
    e4.reset(seed=1)
    e16 = build_env(1, regime="coordination_collective", n_seed_points=16)
    e16.reset(seed=1)
    assert _boundaries(e16.biome_map) > _boundaries(e4.biome_map)


def test_run_overnight_records_n_seed_points(tmp_path):
    from overnight_v8b1 import run_overnight
    rep = run_overnight(n_ticks=20, seed=1, device="cpu", out_dir=str(tmp_path),
                        regime="coordination_collective", n_seed_points=16)
    assert rep["config"]["n_seed_points"] == 16
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_topology_generality.py::test_build_env_propagates_n_seed_points -v`
Expected: FAIL — `TypeError: build_env() got an unexpected keyword argument 'n_seed_points'`

- [ ] **Step 3: Ajouter le param à `build_env`**

Dans `scripts/overnight_v8b1.py`, signature de `build_env` (≈ ligne 65, où se trouve `n_initial_affinities: int = 4,`) — ajouter à la suite :

```python
    n_seed_points: int = 8,
```

- [ ] **Step 4: Passer le param au `BiomeConfig` coordination**

Dans le `biome_cfg = BiomeConfig(...)` du bloc coordination (≈ ligne 200),
remplacer `n_seed_points=8,` par :

```python
            enabled=True, n_seed_points=n_seed_points, balanced_seeds=True,
```

(la ligne actuelle est `enabled=True, n_seed_points=8, balanced_seeds=True,`)

- [ ] **Step 5: Ajouter le param à `run_overnight` + le passer à `build_env`**

Signature de `run_overnight` (≈ ligne 389, où est `n_initial_affinities: int = 4,`) — ajouter :

```python
    n_seed_points: int = 8,
```

Dans l'appel `env = build_env(...)` (≈ ligne 397, qui a déjà `n_initial_affinities=n_initial_affinities,`) — ajouter :

```python
        n_seed_points=n_seed_points,
```

- [ ] **Step 6: Écrire la condition dans le report config**

Dans `final_report["config"]` (≈ ligne 677, où est `"n_initial_affinities": n_initial_affinities,`) — ajouter :

```python
            "n_seed_points": n_seed_points,
```

- [ ] **Step 7: Ajouter le flag CLI + le passer**

Dans `main()` (après le flag `--n-initial-affinities`), ajouter :

```python
    p.add_argument(
        "--n-seed-points", type=int, default=8,
        help="V8-C3 topology — Nb de seeds Voronoi (granularite spatiale). "
             "4=grosses regions, 8=defaut, 16=patchwork.",
    )
```

Et dans l'appel `run_overnight(...)` de `main()` (≈ ligne 940, où est
`n_initial_affinities=args.n_initial_affinities,`) — ajouter :

```python
        n_seed_points=args.n_seed_points,
```

- [ ] **Step 8: Lancer pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_topology_generality.py -q`
Expected: PASS (4 tests). Le test `run_overnight` fait un run CPU 20 ticks.

- [ ] **Step 9: Non-régression globale**

Run: `PYTHONIOENCODING=utf-8 pytest -q`
Expected: tous les tests existants + nouveaux passent (le défaut n_seed_points=8 ne change rien).

- [ ] **Step 10: Commit**

```bash
git add scripts/overnight_v8b1.py tests/test_topology_generality.py
git commit -m "feat(topology): lever n_seed_points {4,8,16} + tracabilite report"
```

---

## Task 2: Runner grille `run_portfolio_topology.ps1`

**Files:**
- Create: `scripts/run_portfolio_topology.ps1`

> Orchestrateur shell. Idempotent : skip si le report existe. Grille
> k{1,4} × n{4,8,16} × seeds. Smoke = 1 cellule CPU rapide.

- [ ] **Step 1: Créer le runner**

Créer `scripts/run_portfolio_topology.ps1` :

```powershell
# Generalite portfolio effect : k {1,4} x n_seed_points {4,8,16} x seeds.
# Idempotent : skip une cellule (seed,k,n) si son report existe deja.
param(
    [int]$Start = 1,
    [int]$End = 8,
    [int[]]$Ks = @(1, 4),
    [int[]]$Ns = @(4, 8, 16),
    [int]$Ticks = 16000,
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"
Set-Location $root
foreach ($n in $Ns) {
    foreach ($k in $Ks) {
        for ($s = $Start; $s -le $End; $s++) {
            $outDir = "results\topo_n${n}_k${k}\seed$s"
            $report = "$outDir\overnight_v8b1_seed$s.json"
            if (Test-Path $report) {
                Write-Host "SKIP seed$s k$k n$n (deja fait)"
                continue
            }
            Write-Host "RUN seed$s k$k n$n $(Get-Date -Format HH:mm:ss)"
            & ".venv\Scripts\python.exe" "scripts\overnight_v8b1.py" `
                --ticks $Ticks --seed $s --device $Device `
                --regime coordination_collective `
                --n-initial-affinities $k `
                --n-seed-points $n `
                --out-dir $outDir
        }
    }
}
Write-Host "TOPOLOGY BATCH DONE $(Get-Date -Format HH:mm:ss)"
```

- [ ] **Step 2: Smoke 1 cellule (CPU rapide)**

Run:
```bash
PYTHONIOENCODING=utf-8 python scripts/overnight_v8b1.py --ticks 40 --seed 1 \
    --device cpu --regime coordination_collective \
    --n-initial-affinities 1 --n-seed-points 16 \
    --out-dir results/topo_smoke/seed1
python -c "import json; r=json.load(open('results/topo_smoke/seed1/overnight_v8b1_seed1.json')); print('k=', r['config']['n_initial_affinities'], 'n=', r['config']['n_seed_points'])"
```
Expected: `k= 1 n= 16` (les deux flags traversent jusqu'au report).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_portfolio_topology.ps1
git commit -m "feat(topology): runner grille k{1,4} x n_seed_points{4,8,16}"
```

---

## Task 3: `aggregate_topology.py` — grille extinction + verdict

**Files:**
- Create: `scripts/aggregate_topology.py`
- Test: `tests/test_topology_generality.py`

> Lit les reports overnight, groupe par (k, n_seed_points), sort le taux
> d'extinction par cellule + n_alive/gather moyens (garde-fous), et le verdict
> H_spatial (k=1 ext chute avec n) vs H_type (reste haute).

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_topology_generality.py` :

```python
def _fake_report(seed, k, n, n_alive, gather):
    return {
        "config": {"seed": seed, "n_initial_affinities": k, "n_seed_points": n},
        "final_state": {"n_alive": n_alive},
        "cooperative_v8c3": {"gather_successes_total": gather},
    }


def test_extract_topology_row():
    from aggregate_topology import extract_topo
    r = extract_topo(_fake_report(1, 1, 16, 0, 0))
    assert r["seed"] == 1 and r["k"] == 1 and r["n"] == 16
    assert r["extinct"] is True and r["n_alive"] == 0


def test_summarize_topology_grid():
    from aggregate_topology import summarize_topo
    rows = [
        _row(1, 1, 4, 0), _row(2, 1, 4, 0),       # k=1 n=4 : 2/2 eteints
        _row(1, 1, 16, 60), _row(2, 1, 16, 58),   # k=1 n=16 : 0/2 eteints
        _row(1, 4, 4, 61), _row(1, 4, 16, 62),    # k=4 : survit
    ]
    s = summarize_topo(rows)
    assert s["grid"][(1, 4)]["extinction_pct"] == 100
    assert s["grid"][(1, 16)]["extinction_pct"] == 0
    assert s["grid"][(4, 4)]["extinction_pct"] == 0


def _row(seed, k, n, n_alive):
    from aggregate_topology import extract_topo
    return extract_topo(_fake_report(seed, k, n, n_alive, 50))
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_topology_generality.py::test_extract_topology_row -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aggregate_topology'`

- [ ] **Step 3: Implémenter `aggregate_topology.py`**

Créer `scripts/aggregate_topology.py` :

```python
"""Agrégation — généralité du portfolio effect par topologie spatiale.

Grille extinction par (k, n_seed_points). Tranche H_spatial (k=1 ext chute quand
n_seed_points augmente -> fragmenter l'espace sauve la monoculture) vs H_type
(k=1 reste fragile a tout n).

Usage:
    python scripts/aggregate_topology.py results/topo_n4_k1/seed* \\
        results/topo_n4_k4/seed* results/topo_n8_k1/seed* ...
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from typing import Any


def extract_topo(report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("config", {})
    fs = report.get("final_state", {}) or {}
    coop = report.get("cooperative_v8c3", {}) or {}
    n_alive = fs.get("n_alive", 0) or 0
    return {
        "seed": cfg.get("seed"),
        "k": cfg.get("n_initial_affinities"),
        "n": cfg.get("n_seed_points"),
        "n_alive": n_alive,
        "gather": coop.get("gather_successes_total", 0) or 0,
        "extinct": n_alive == 0,
    }


def summarize_topo(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple, list] = {}
    for r in rows:
        cells.setdefault((r["k"], r["n"]), []).append(r)
    grid: dict[tuple, dict] = {}
    for key, rs in cells.items():
        grid[key] = {
            "n": len(rs),
            "extinction_pct": round(100 * sum(x["extinct"] for x in rs) / len(rs)),
            "alive_mean": round(st.mean(x["n_alive"] for x in rs), 1),
            "gather_mean": round(st.mean(x["gather"] for x in rs), 1),
        }
    return {"grid": grid}


def main() -> None:
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: aggregate_topology.py <dir> [dir ...]")
        sys.exit(1)
    rows = []
    for d in dirs:
        for f in glob.glob(f"{d}/overnight_v8b1_seed*.json"):
            with open(f, encoding="utf-8") as fh:
                rows.append(extract_topo(json.load(fh)))
    s = summarize_topo(rows)
    ns = sorted({k[1] for k in s["grid"]})
    ks = sorted({k[0] for k in s["grid"]})
    print(f"n_seed_points : " + "  ".join(f"{n:>8}" for n in ns))
    for k in ks:
        cells = [s["grid"].get((k, n)) for n in ns]
        ext = "  ".join(
            f"{c['extinction_pct']:>3}%({c['n']})" if c else "   --   "
            for c in cells)
        print(f"  k={k} extinction : {ext}")
    for k in ks:
        cells = [s["grid"].get((k, n)) for n in ns]
        al = "  ".join(f"{c['alive_mean']:>8}" if c else "   --   "
                       for c in cells)
        print(f"  k={k} alive_moy  : {al}")
    # verdict H_spatial vs H_type (sur k=1)
    k1 = [(n, s["grid"].get((1, n))) for n in ns]
    k1 = [(n, c["extinction_pct"]) for n, c in k1 if c]
    if len(k1) >= 2:
        first, last = k1[0][1], k1[-1][1]
        verdict = ("H_spatial (fragmenter sauve la monoculture)"
                   if last < first - 15 else
                   "H_type (k=1 reste fragile)" if last > first - 15
                   else "ambigu")
        print(f"\n--- k=1 extinction {k1[0][0]}->{k1[-1][0]} seeds : "
              f"{first}% -> {last}%  => {verdict} ---")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `pytest tests/test_topology_generality.py -q`
Expected: PASS (tous : config + reset + agrégation).

- [ ] **Step 5: Commit**

```bash
git add scripts/aggregate_topology.py tests/test_topology_generality.py
git commit -m "feat(topology): aggregate_topology — grille extinction + verdict H_spatial/H_type"
```

---

## Task 4: Vérification finale + batch + finding

**Files:** aucun nouveau (validation + run GPU).

- [ ] **Step 1: Suite complète verte**

Run: `PYTHONIOENCODING=utf-8 pytest -q`
Expected: tous verts (existants + topology), zéro régression.

- [ ] **Step 2: Lancer le batch (GPU, ~13h N=8, en arrière-plan)**

Run (PowerShell, arrière-plan) :
```powershell
& "scripts\run_portfolio_topology.ps1" -Start 1 -End 8
```
48 runs (k{1,4} × n{4,8,16} × 8 seeds), idempotent. Sorties :
`results/topo_n{4,8,16}_k{1,4}/seed{1..8}/`. (Réduire `-End 6` pour 36 runs ~10h.)

- [ ] **Step 3: Agréger après le batch**

Run:
```bash
python scripts/aggregate_topology.py results/topo_n4_k1/seed* results/topo_n4_k4/seed* \
    results/topo_n8_k1/seed* results/topo_n8_k4/seed* \
    results/topo_n16_k1/seed* results/topo_n16_k4/seed*
```
Lire la grille d'extinction. **k=1 ext chute avec n → H_spatial** (fragmenter sauve
la monoculture, le portfolio est spatial/général). **k=1 reste haute → H_type.**
Vérifier k=4 robuste partout (contrôle) + garde-fou gather (survie non moribonde).

- [ ] **Step 4: Écrire le finding**

Selon le résultat, écrire
`docs/findings/2026-06-XX-finding-v8c3-portfolio-generality-topology.md`
(H_spatial : portfolio général / H_type : type-dépendant) + mettre à jour le
quasi-paper (section généralité) + SYNTHESIS. Commit.

---

## Self-Review (auteur du plan)

- **Couverture spec** : lever n_seed_points build_env/run_overnight/CLI + report (T1) ✓ · reset produit topologies différentes (T1 test fragmentation) ✓ · non-régression défaut 8 (T1 step 9) ✓ · runner grille k×n (T2) ✓ · aggregate grille extinction + verdict H_spatial/H_type (T3) ✓ · garde-fous alive/gather (T3) ✓ · batch + finding (T4) ✓ · k={1,4} extrêmes ✓.
- **Écarts assumés** : comptage explicite des patches/réservoirs omis (la grille d'extinction tranche, cf. spec §5/§10). Seuil verdict `last < first - 15` = heuristique (le finding lira les chiffres bruts). N ajustable au runner (`-End`).
- **Pas de placeholder** : code complet.
- **Cohérence types** : `n_seed_points` ajouté aux mêmes endroits que `n_initial_affinities` (vérifié lignes 65/200/389/397/677/940) ; `extract_topo(report)->dict{seed,k,n,n_alive,gather,extinct}` et `summarize_topo(rows)->{grid:{(k,n):{...}}}` cohérents T3 ; clé report `config.n_seed_points` identique entre écriture (T1) et lecture (T3).
