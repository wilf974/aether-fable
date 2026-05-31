# C2 — Affinity Diversity Causal Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tester causalement si la diversité d'affinité initiale (`n_initial_affinities ∈ {1,2,4}`) détermine la mobilité spatiale (`mobility_score`), via un design apparié 10 seeds.

**Architecture:** Un flag `n_initial_affinities` (défaut 4 = actuel) ajouté à `BiomeConfig`, lu par `seasonal_grid.reset()` (`agent_id % k` au lieu de `% 4`), plombé via `build_env`/`run_overnight`/CLI. Un runner PowerShell lance 10 seeds × 3 conditions ; `aggregate_c2.py` produit les deltas intra-seed.

**Tech Stack:** Python 3.13, numpy, pytest. Runs overnight CUDA (batch séparé, hors TDD).

**Spec :** `docs/superpowers/specs/2026-05-31-c2-affinity-diversity-causal-design.md`

---

## File Structure

| Fichier | Responsabilité |
|---|---|
| `aetherlife/world/biomes.py` | `+ BiomeConfig.n_initial_affinities` + validation |
| `aetherlife/world/seasonal_grid.py` | `reset()` : assignation affinité `% n_initial_affinities` |
| `scripts/overnight_v8b1.py` | plomberie `build_env`/`run_overnight`/CLI + traçabilité report |
| `scripts/run_c2_affinity.ps1` | runner batch 10 seeds × {1,2,4}, idempotent |
| `scripts/aggregate_c2.py` | extraction + résumé apparié (deltas intra-seed) |
| `tests/test_c2_affinity.py` | tests config + reset + traçabilité + agrégation |

---

## Task 1: `BiomeConfig.n_initial_affinities` + validation

**Files:**
- Modify: `aetherlife/world/biomes.py` (champ ~ligne 109, validation dans `__post_init__` ~ligne 130)
- Test: `tests/test_c2_affinity.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_c2_affinity.py` :

```python
import pytest

from aetherlife.world.biomes import BiomeConfig


def test_n_initial_affinities_default_is_4():
    assert BiomeConfig().n_initial_affinities == 4


def test_n_initial_affinities_accepts_1_2_4():
    for k in (1, 2, 4):
        assert BiomeConfig(n_initial_affinities=k).n_initial_affinities == k


def test_n_initial_affinities_rejects_zero():
    with pytest.raises(ValueError):
        BiomeConfig(n_initial_affinities=0)


def test_n_initial_affinities_rejects_above_4():
    with pytest.raises(ValueError):
        BiomeConfig(n_initial_affinities=5)
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_c2_affinity.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'n_initial_affinities'`

- [ ] **Step 3: Ajouter le champ**

Dans `aetherlife/world/biomes.py`, après la ligne
`reproduction_locked_to_affinity: bool = True` (≈ ligne 109), ajouter :

```python
    # V8-C3 C2 — nb d'affinités assignées aux fondateurs (round-robin).
    # Défaut 4 = comportement historique (5/5/5/5 sur 20 agents). 1 = mono.
    n_initial_affinities: int = 4
```

- [ ] **Step 4: Ajouter la validation**

Dans `BiomeConfig.__post_init__` (après la validation `passage_width`, ≈ ligne 140), ajouter :

```python
        if not (1 <= self.n_initial_affinities <= 4):
            raise ValueError(
                f"n_initial_affinities doit être dans [1, 4] "
                f"(got {self.n_initial_affinities})"
            )
```

- [ ] **Step 5: Lancer pour vérifier le succès**

Run: `pytest tests/test_c2_affinity.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add aetherlife/world/biomes.py tests/test_c2_affinity.py
git commit -m "feat(biomes): BiomeConfig.n_initial_affinities + validation"
```

---

## Task 2: Plomberie `build_env` / `run_overnight` / CLI + traçabilité

**Files:**
- Modify: `scripts/overnight_v8b1.py` (`build_env` ~56-62 + BiomeConfig coordination ~189-208 ; `run_overnight` ~370-375 + report config ~642-645 ; `main` argparse ~907-921)
- Test: `tests/test_c2_affinity.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_c2_affinity.py` :

```python
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)


def test_build_env_propagates_n_initial_affinities():
    from overnight_v8b1 import build_env
    env = build_env(seed=1, regime="coordination_collective",
                    n_initial_affinities=2)
    assert env.cfg.biomes.n_initial_affinities == 2


def test_build_env_defaults_to_4():
    from overnight_v8b1 import build_env
    env = build_env(seed=1, regime="coordination_collective")
    assert env.cfg.biomes.n_initial_affinities == 4


def test_run_overnight_records_condition_in_report(tmp_path):
    from overnight_v8b1 import run_overnight
    report = run_overnight(
        n_ticks=20, seed=1, device="cpu", out_dir=str(tmp_path),
        regime="coordination_collective", n_initial_affinities=2,
    )
    assert report["config"]["n_initial_affinities"] == 2
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_c2_affinity.py::test_build_env_propagates_n_initial_affinities -v`
Expected: FAIL — `TypeError: build_env() got an unexpected keyword argument 'n_initial_affinities'`

- [ ] **Step 3: Ajouter le param à `build_env`**

Dans `scripts/overnight_v8b1.py`, signature de `build_env` (≈ ligne 56) — ajouter le paramètre :

```python
def build_env(
    seed: int, *, regime: str = "training",
    disable_vocalize_after_tick: int | None = None,
    vocalize_energy_cost: float = 0.05,
    max_pop_override: int | None = None,
    bonus_energy_override: float | None = None,
    n_initial_affinities: int = 4,
) -> SeasonalMultiAgentFoodGrid:
```

- [ ] **Step 4: Passer le param au `BiomeConfig` du régime coordination**

Dans `build_env`, le `biome_cfg = BiomeConfig(...)` du bloc coordination (≈ ligne 189), ajouter le champ `n_initial_affinities=n_initial_affinities` :

```python
        biome_cfg = BiomeConfig(
            enabled=True, n_seed_points=8, balanced_seeds=True,
            affinity_enabled=True,
            in_affinity_metabolism=0.7, in_affinity_food_value=1.3,
            out_affinity_metabolism=1.5, out_affinity_food_value=0.7,
            out_affinity_movement_mult=2.5,
            reproduction_locked_to_affinity=True,
            respawn_enabled=True,
            respawn_check_every=200,
            respawn_extinct_after_ticks=3000,
            respawn_threshold=respawn_thr,
            respawn_initial_energy=200.0,
            seed_bank_max_per_affinity=2,
            hidden_food=(regime in (
                "coordination_hidden", "coordination_hard",
            )),
            n_initial_affinities=n_initial_affinities,
        )
```

- [ ] **Step 5: Ajouter le param à `run_overnight` + le passer à `build_env`**

Signature de `run_overnight` (≈ ligne 370) — ajouter `n_initial_affinities: int = 4` :

```python
def run_overnight(
    n_ticks: int, seed: int, device: str, out_dir: str,
    snap_every: int = 5000, divergence_every: int = 5000,
    regime: str = "training",
    disable_vocalize_after_tick: int | None = None,
    vocalize_energy_cost: float = 0.05,
    max_pop_override: int | None = None,
    bonus_energy_override: float | None = None,
    n_initial_affinities: int = 4,
) -> dict:
```

Dans `run_overnight`, l'appel `env = build_env(...)` (≈ ligne 376) — ajouter le param :

```python
    env = build_env(
        seed, regime=regime,
        disable_vocalize_after_tick=disable_vocalize_after_tick,
        vocalize_energy_cost=vocalize_energy_cost,
        max_pop_override=max_pop_override,
        bonus_energy_override=bonus_energy_override,
        n_initial_affinities=n_initial_affinities,
    )
```

- [ ] **Step 6: Écrire la condition dans le report config**

Dans `final_report["config"]` (≈ ligne 642), ajouter la clé :

```python
        "config": {
            "n_ticks": n_ticks, "seed": seed, "device": device,
            "obs_dim": policy.obs_dim, "vision_radius": cfg.vision_radius,
            "n_initial_affinities": n_initial_affinities,
        },
```

- [ ] **Step 7: Ajouter le flag CLI**

Dans `main()` (≈ ligne 911, après `--bonus-energy-override`), ajouter :

```python
    p.add_argument(
        "--n-initial-affinities", type=int, default=4,
        help="V8-C3 C2 — Nb d'affinités assignées aux fondateurs (1=mono, "
             "4=multi/défaut). Test causal diversité d'affinité.",
    )
```

Et dans l'appel `run_overnight(...)` de `main()` (≈ ligne 913), ajouter :

```python
        n_initial_affinities=args.n_initial_affinities,
```

- [ ] **Step 8: Lancer pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_c2_affinity.py -v`
Expected: PASS (7 tests — 4 de Task 1 + 3 ici). Le test `run_overnight` fait un run CPU 20 ticks (quelques secondes).

- [ ] **Step 9: Commit**

```bash
git add scripts/overnight_v8b1.py tests/test_c2_affinity.py
git commit -m "feat(overnight): plomberie n_initial_affinities + tracabilite report"
```

---

## Task 3: `reset()` applique `n_initial_affinities`

**Files:**
- Modify: `aetherlife/world/seasonal_grid.py` (≈ ligne 552)
- Test: `tests/test_c2_affinity.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_c2_affinity.py` :

```python
from collections import Counter


def _affinities(k):
    from overnight_v8b1 import build_env
    env = build_env(seed=1, regime="coordination_collective",
                    n_initial_affinities=k)
    env.reset(seed=1)
    return Counter(a.biome_affinity for a in env._agents)  # noqa: SLF001


def test_reset_k1_all_affinity_zero():
    assert set(_affinities(1)) == {0}


def test_reset_k2_two_affinities_balanced():
    c = _affinities(2)
    assert set(c) == {0, 1}
    assert c[0] == 10 and c[1] == 10  # 20 agents, round-robin %2


def test_reset_k4_balanced_5_each_nonregression():
    assert dict(_affinities(4)) == {0: 5, 1: 5, 2: 5, 3: 5}
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_c2_affinity.py::test_reset_k1_all_affinity_zero -v`
Expected: FAIL — `assert set(...) == {0}` échoue (k=1 donne encore {0,1,2,3} car `% 4` est codé en dur).

- [ ] **Step 3: Modifier l'assignation d'affinité**

Dans `aetherlife/world/seasonal_grid.py`, `reset()` (≈ ligne 552), remplacer :

```python
                # Distribution uniforme via round-robin sur agent_id
                a.biome_affinity = a.agent_id % 4
```

par :

```python
                # V8-C3 C2 — round-robin sur n_initial_affinities (défaut 4).
                # 1 = mono-affinité (tous biome 0), 4 = multi équilibré.
                a.biome_affinity = a.agent_id % bcfg.n_initial_affinities
```

(`bcfg = self.cfg.biomes` est déjà défini ≈ ligne 533.)

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_c2_affinity.py -v`
Expected: PASS (10 tests). Le `test_reset_k4...` garantit la **non-régression** (5/5/5/5 inchangé).

- [ ] **Step 5: Non-régression globale**

Run: `PYTHONIOENCODING=utf-8 pytest -q`
Expected: tous les tests existants + nouveaux passent (le défaut k=4 ne change rien).

- [ ] **Step 6: Commit**

```bash
git add aetherlife/world/seasonal_grid.py tests/test_c2_affinity.py
git commit -m "feat(seasonal_grid): affinite fondateurs % n_initial_affinities (C2)"
```

---

## Task 4: Runner batch `run_c2_affinity.ps1`

**Files:**
- Create: `scripts/run_c2_affinity.ps1`

> Orchestrateur shell (pas de test unitaire). Idempotent : skip si le report
> existe déjà. Vérifié en lançant 1 cellule (seed1, k=4) en smoke.

- [ ] **Step 1: Créer le runner**

Créer `scripts/run_c2_affinity.ps1` :

```powershell
# C2 — test causal diversite d'affinite. 10 seeds x {1,2,4}, design apparie.
# Idempotent : skip un (seed,k) si son report existe deja.
param(
    [int]$Start = 1,
    [int]$End = 10,
    [int[]]$Ks = @(1, 2, 4),
    [int]$Ticks = 16000,
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"
Set-Location $root
foreach ($k in $Ks) {
    for ($s = $Start; $s -le $End; $s++) {
        $outDir = "results\c2_aff$k\seed$s"
        $report = "$outDir\overnight_v8b1_seed$s.json"
        if (Test-Path $report) {
            Write-Host "SKIP seed$s k$k (deja fait)"
            continue
        }
        Write-Host "RUN seed$s k$k $(Get-Date -Format HH:mm:ss)"
        & ".venv\Scripts\python.exe" "scripts\overnight_v8b1.py" `
            --ticks $Ticks --seed $s --device $Device `
            --regime coordination_collective `
            --n-initial-affinities $k `
            --out-dir $outDir
    }
}
Write-Host "C2 BATCH DONE $(Get-Date -Format HH:mm:ss)"
```

- [ ] **Step 2: Smoke 1 cellule (CPU rapide)**

Run:
```bash
PYTHONIOENCODING=utf-8 python scripts/overnight_v8b1.py --ticks 40 --seed 1 \
    --device cpu --regime coordination_collective --n-initial-affinities 1 \
    --out-dir results/c2_smoke/seed1
python -c "import json; r=json.load(open('results/c2_smoke/seed1/overnight_v8b1_seed1.json')); print('cond=', r['config']['n_initial_affinities'])"
```
Expected: `cond= 1` (le flag CLI traverse bien jusqu'au report).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_c2_affinity.ps1
git commit -m "feat(scripts): runner C2 affinity diversity (10 seeds x {1,2,4})"
```

---

## Task 5: `aggregate_c2.py` — résumé apparié

**Files:**
- Create: `scripts/aggregate_c2.py`
- Test: `tests/test_c2_affinity.py`

> Métriques tirées du report overnight : `mobility_score`, `village_basin`,
> `n_alive`, `gather_successes`, `extinction`, `aff_conc_final` (depuis
> `final_state.affinity_distribution`). **`occ_biome_conc` OMIS** : nécessite les
> positions par tick (recorder), absentes du report overnight.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_c2_affinity.py` :

```python
def _fake_report(seed, k, mobility, n_alive, gather, aff_dist):
    return {
        "config": {"seed": seed, "n_initial_affinities": k},
        "spatial_mobility_v8c3": {
            "corr_occupation_start_end": mobility,
            "village_basin": (mobility is not None and mobility >= 0.8),
        },
        "final_state": {"n_alive": n_alive, "affinity_distribution": aff_dist},
        "cooperative_v8c3": {"gather_successes_total": gather},
    }


def test_extract_c2_row():
    from aggregate_c2 import extract_c2
    r = extract_c2(_fake_report(1, 1, 0.9, 60, 120, {"0": 60}))
    assert r["seed"] == 1 and r["k"] == 1
    assert r["mobility_score"] == 0.9 and r["village_basin"] is True
    assert r["n_alive"] == 60 and r["gather_successes"] == 120
    assert r["extinction"] is False
    assert r["aff_conc_final"] == 1.0  # 60/60


def test_extract_c2_extinction_and_affconc():
    from aggregate_c2 import extract_c2
    r = extract_c2(_fake_report(2, 4, None, 0, 0, {"0": 0}))
    assert r["extinction"] is True
    assert r["aff_conc_final"] == 0.0  # population vide


def test_summarize_c2_paired_delta_and_sign():
    from aggregate_c2 import summarize_c2
    rows = [
        extract_dict(1, 1, 0.90), extract_dict(1, 4, 0.40),
        extract_dict(2, 1, 0.85), extract_dict(2, 4, 0.50),
        extract_dict(3, 1, 0.30), extract_dict(3, 4, 0.60),  # contre-exemple
    ]
    summary = summarize_c2(rows)
    # delta intra-seed k1-k4
    assert summary["paired"][1]["delta_k1_k4"] == pytest.approx(0.50)
    # 2/3 seeds ont k1 > k4
    assert summary["n_seeds_k1_gt_k4"] == 2
    assert summary["n_paired"] == 3


def extract_dict(seed, k, mobility):
    from aggregate_c2 import extract_c2
    return extract_c2(_fake_report(seed, k, mobility, 60, 100, {"0": 50, "1": 10}))
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_c2_affinity.py::test_extract_c2_row -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aggregate_c2'`

- [ ] **Step 3: Implémenter `aggregate_c2.py`**

Créer `scripts/aggregate_c2.py` :

```python
"""Agrégation C2 — diversité d'affinité → mobilité (design apparié).

Lit les reports overnight de results/c2_aff{k}/seed{s}/ et produit :
- une table appariée par seed (mobility k1/k2/k4 + delta intra-seed)
- les moyennes par condition + garde-fous survie
- le test du signe apparié (nb de seeds où mobility_k1 > mobility_k4)

Usage:
    python scripts/aggregate_c2.py results/c2_aff1/seed* results/c2_aff2/seed* \\
        results/c2_aff4/seed*
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from typing import Any


def extract_c2(report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("config", {})
    sm = report.get("spatial_mobility_v8c3", {}) or {}
    fs = report.get("final_state", {}) or {}
    coop = report.get("cooperative_v8c3", {}) or {}
    aff_dist = fs.get("affinity_distribution", {}) or {}
    counts = [int(v) for v in aff_dist.values()]
    total = sum(counts)
    aff_conc = (max(counts) / total) if total > 0 else 0.0
    n_alive = fs.get("n_alive", 0) or 0
    return {
        "seed": cfg.get("seed"),
        "k": cfg.get("n_initial_affinities"),
        "mobility_score": sm.get("corr_occupation_start_end"),
        "village_basin": sm.get("village_basin"),
        "n_alive": n_alive,
        "gather_successes": coop.get("gather_successes_total", 0) or 0,
        "extinction": n_alive == 0,
        "aff_conc_final": round(aff_conc, 3),
    }


def summarize_c2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[Any, dict[int, dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["k"]] = r
    paired: dict[Any, dict] = {}
    n_k1_gt_k4 = 0
    n_paired = 0
    for seed, byk in by_seed.items():
        if 1 in byk and 4 in byk:
            m1 = byk[1]["mobility_score"]
            m4 = byk[4]["mobility_score"]
            if m1 is not None and m4 is not None:
                n_paired += 1
                delta = m1 - m4
                paired[seed] = {"mobility_k1": m1, "mobility_k4": m4,
                                "delta_k1_k4": delta}
                if m1 > m4:
                    n_k1_gt_k4 += 1
    by_cond: dict[int, dict] = {}
    for k in (1, 2, 4):
        ms = [r["mobility_score"] for r in rows
              if r["k"] == k and r["mobility_score"] is not None]
        vb = [r["village_basin"] for r in rows
              if r["k"] == k and r["village_basin"] is not None]
        alive = [r["n_alive"] for r in rows if r["k"] == k]
        gather = [r["gather_successes"] for r in rows if r["k"] == k]
        ext = [r["extinction"] for r in rows if r["k"] == k]
        if ms:
            by_cond[k] = {
                "mobility_mean": round(st.mean(ms), 3),
                "village_pct": round(100 * sum(vb) / len(vb)) if vb else None,
                "alive_mean": round(st.mean(alive), 1) if alive else None,
                "gather_mean": round(st.mean(gather), 1) if gather else None,
                "extinction_pct": round(100 * sum(ext) / len(ext)) if ext else None,
                "n": len(ms),
            }
    return {
        "paired": paired, "by_cond": by_cond,
        "n_paired": n_paired, "n_seeds_k1_gt_k4": n_k1_gt_k4,
    }


def main() -> None:
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: aggregate_c2.py <clip_dir> [clip_dir ...]")
        sys.exit(1)
    rows = []
    for d in dirs:
        for f in glob.glob(f"{d}/overnight_v8b1_seed*.json"):
            with open(f, encoding="utf-8") as fh:
                rows.append(extract_c2(json.load(fh)))
    s = summarize_c2(rows)
    print(f"{'seed':>5} {'mob_k1':>7} {'mob_k4':>7} {'Δ(k1-k4)':>9}")
    for seed, p in sorted(s["paired"].items()):
        print(f"{str(seed):>5} {p['mobility_k1']:>7.3f} "
              f"{p['mobility_k4']:>7.3f} {p['delta_k1_k4']:>+9.3f}")
    print(f"\n--- {s['n_seeds_k1_gt_k4']}/{s['n_paired']} seeds : "
          f"mobility_k1 > mobility_k4 (test du signe apparié) ---")
    print(f"{'k':>3} {'mob':>6} {'village%':>8} {'alive':>6} "
          f"{'gather':>7} {'ext%':>5}")
    for k in (1, 2, 4):
        c = s["by_cond"].get(k)
        if c:
            print(f"{k:>3} {c['mobility_mean']:>6.3f} "
                  f"{str(c['village_pct']):>8} {str(c['alive_mean']):>6} "
                  f"{str(c['gather_mean']):>7} {str(c['extinction_pct']):>5}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `pytest tests/test_c2_affinity.py -v`
Expected: PASS (tous : config + plomberie + reset + agrégation).

- [ ] **Step 5: Commit**

```bash
git add scripts/aggregate_c2.py tests/test_c2_affinity.py
git commit -m "feat(scripts): aggregate_c2 — resume apparie + deltas intra-seed"
```

---

## Task 6: Vérification finale + lancement du batch

**Files:** aucun nouveau (validation + run GPU).

- [ ] **Step 1: Suite complète verte**

Run: `PYTHONIOENCODING=utf-8 pytest -q`
Expected: tous verts (existants + ~13 nouveaux C2), zéro régression.

- [ ] **Step 2: Lancer le batch (GPU, ~8h, en arrière-plan)**

Run (PowerShell, arrière-plan) :
```powershell
& "scripts\run_c2_affinity.ps1" -Start 1 -End 10
```
30 runs (10 seeds × {1,2,4}), idempotent. Sorties : `results/c2_aff{1,2,4}/seed{1..10}/`.

- [ ] **Step 3: Agréger après le batch**

Run:
```bash
python scripts/aggregate_c2.py results/c2_aff1/seed* results/c2_aff2/seed* \
    results/c2_aff4/seed*
```
Lire : `mobility_k1 > mobility_k4` sur la majorité des seeds appariés ? Survie
comparable (garde-fous §4 spec) ? → verdict causal.

- [ ] **Step 4: Écrire le finding C2**

Selon le résultat (causal confirmé / réfuté / ambigu), écrire
`docs/findings/2026-05-XX-finding-v8c3-c2-affinity-causal.md` + commit.

---

## Self-Review (auteur du plan)

- **Couverture spec** : flag BiomeConfig+validation (T1) ✓ · reset % k (T3) ✓ · plomberie+traçabilité (T2) ✓ · runner apparié (T4) ✓ · aggregate deltas intra-seed (T5) ✓ · garde-fous survie dans aggregate (alive/gather/ext) ✓ · non-régression k=4 (T3 step) ✓.
- **Écart spec assumé** : `occ_biome_conc` omis de l'agrégat (pas dans le report overnight — positions absentes). `aff_conc_final` le remplace pour vérifier le mécanisme. Acté en tête de T5.
- **Pas de placeholder** : tout le code est complet.
- **Cohérence des types** : `extract_c2(report)->dict` et `summarize_c2(rows)->dict` cohérents T5 ; clé `n_initial_affinities` identique dans build_env/run_overnight/report config/CLI ; `bcfg.n_initial_affinities` lu en T3 == champ défini T1.
