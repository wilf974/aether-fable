# Spec — Généralité du Portfolio Effect : topologie spatiale (n_seed_points)

**Date** : 2026-06-06
**Sous-projet** : V8-C3 — test de généralité du résultat phare (effet portefeuille).
**Statut** : design validé, prêt pour plan TDD.
**Branche cible** : `master` (via `feature/portfolio-generality-topology`).

---

## 1. Objectif

L'effet portefeuille (diversité d'affinité → survie, extinction 60/30/10 % pour
k=1/2/4) est le résultat causal le plus robuste d'AetherLife. Question :
**est-ce une propriété du système ou un artefact du monde Voronoi à 8 seeds / 4 biomes ?**

On varie la **granularité spatiale** de la partition Voronoi (`n_seed_points`) et on
re-joue l'intervention de survie (k=1 vs k=4). C'est aussi un **test de mécanisme** :
il découple la diversité de TYPES de la diversité SPATIALE.

> **Question décisive** : *fragmenter l'espace suffit-il à sauver la monoculture ?*

## 2. Hypothèses (mutuellement exclusives sur la lecture)

À `n_seed_points=16`, une population mono-affinité (k=1) trouve sa **seule** affinité
éclatée en ~4 patches dispersés → plusieurs réservoirs **spatiaux** malgré un seul type.

| Lecture | Si on observe… | Conclusion |
|---|---|---|
| **H_spatial** | extinction k=1 **chute** quand n_seed_points ↑ (4→16) | le tampon est **spatial** → effet plus général que « plusieurs affinités » |
| **H_type** | extinction k=1 **reste haute** à tout n | la diversité de **types** est nécessaire |
| combiné | k=4 **robuste partout** + k=1 sensible à n | type et spatialité se combinent |

**Le résultat survie SEUL tranche** : l'amélioration (ou non) de la survie de k=1 avec
n_seed_points *est* la preuve directe (pas besoin de compter les patches).

## 3. Manipulation (1 lever additif)

Exposer `n_seed_points` dans `build_env` (figé à 8 dans le `BiomeConfig` du régime
coordination). Pattern identique à `n_initial_affinities` (C2). Défaut **8** =
rétro-compatible. `balanced_seeds=True` conservé (chaque type d'affinité présent).

## 4. Plan expérimental — grille 2×3

```
k ∈ {1, 4}   ×   n_seed_points ∈ {4, 8, 16}   ×   N seeds (défaut 8)
```
- **Apparié** : pour chaque (seed, n_seed_points), k=1 vs k=4 sur le **même
  biome_map** (déterministe par (seed, n)). `n_seed_points` est l'axe
  entre-conditions (les maps diffèrent par construction — c'est la manipulation).
- Runner = **overnight** (l'extinction est rapportée proprement dans le report,
  comme C2). N ajustable au lancement.
- Coût : 2×3×8 = **48 runs overnight** (~13h ; N=6 → 36 runs ~10h).

## 5. Métriques

- **Primaire — extinction par cellule (k, n_seed_points)** : `n_alive == 0` en fin
  de run (depuis `final_state.n_alive`).
- **Garde-fous** (lire survie ET vitalité, leçon C2 §4) : `n_alive` moyen, gather
  moyen — pour ne pas lire « k=1 survit à n=16 » s'il survit moribond.
- **Mécanisme (confirmatoire)** : `spatial_mobility_v8c3` déjà dans le report ; pas
  de comptage de patches au MVP (la grille d'extinction suffit à trancher §2).

## 6. Plomberie (additive, rétro-compatible)

| Fichier | Changement |
|---|---|
| `scripts/overnight_v8b1.py` (`build_env`) | `+ n_seed_points: int = 8` → `BiomeConfig(n_seed_points=n_seed_points, ...)` du régime coordination |
| `scripts/overnight_v8b1.py` (`run_overnight`) | `+ n_seed_points` → build_env + écrit `report["config"]["n_seed_points"]` (traçabilité) |
| `scripts/overnight_v8b1.py` (CLI) | `+ --n-seed-points` (défaut 8) |
| `scripts/run_portfolio_topology.ps1` | runner grille k{1,4} × n{4,8,16} × seeds, idempotent |
| `scripts/aggregate_topology.py` | extraction + grille extinction par (k, n_seed_points) + lecture H_spatial/H_type |

**`build_env`/`run_overnight` gardent leur signature actuelle** (params ajoutés avec
défaut 8) → 0 régression sur les runs existants.

## 7. `aggregate_topology.py` — sortie

```
n_seed_points :     4         8        16
  k=1 extinction :  X/N      X/N      X/N      <- chute = H_spatial
  k=4 extinction :  X/N      X/N      X/N      <- doit rester bas (controle)
  k=1 alive_moy  :  ...
--- VERDICT : k=1 ext chute avec n ? (H_spatial) | reste haute ? (H_type) ---
```

## 8. Garde-fous d'interprétation

1. Lire **survie ET vitalité** : k=1 « survivant » à n=16 mais avec gather effondré
   = survie moribonde, pas un vrai sauvetage spatial.
2. `n_seed_points=4` avec `balanced_seeds=True` → exactement 4 patches (1/type). C'est
   le cas le plus concentré ; le baseline est n=8 (l'actuel).
3. Les maps diffèrent forcément entre n — c'est la manipulation, pas un confond.

## 9. Tests (TDD, sans GPU)

| Test | Vérifie |
|---|---|
| `build_env(n_seed_points=4)` → `env.cfg.biomes.n_seed_points == 4` | propagation |
| `build_env()` défaut → `n_seed_points == 8` | rétro-compat |
| `reset()` avec n_seed_points=4 vs 16 → biome_map a des structures différentes (nb de régions distinctes croît) | manipulation effective |
| `run_overnight(n_seed_points=4)` → `report["config"]["n_seed_points"] == 4` | traçabilité |
| `aggregate_topology.extract` sur report synthétique → cellule (k, n) correcte | agrégation |

Le gros est testable sans GPU (config + reset + agrégation). Le batch = GPU séparé.

## 10. Hors scope

- `worldgen="continental"` (non implémenté ; serait un build worldgen séparé).
- `balanced_seeds=False` (déséquilibre de biomes — autre axe, confond).
- Comptage explicite des patches/réservoirs spatiaux (la grille d'extinction tranche ;
  re-record ciblé possible en suivi si le résultat est ambigu).
- k=2 (les extrêmes k=1/4 suffisent au contraste décisif).

## 11. Livrables

- `build_env`/`run_overnight`/CLI : lever `n_seed_points` + traçabilité
- `scripts/run_portfolio_topology.ps1` (runner grille)
- `scripts/aggregate_topology.py` (grille extinction + verdict)
- `tests/test_topology_generality.py` (config + reset + agrégation, sans GPU)
- Finding après batch : portfolio général (H_spatial) ou type-dépendant (H_type).
