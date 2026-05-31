# Spec — C2 : test causal de la diversité d'affinité → mobilité spatiale

**Date** : 2026-05-31
**Sous-projet** : V8-C3, chantier C2 (test causal de la chaîne révélée par C0).
**Statut** : design validé, prêt pour plan TDD.
**Branche cible** : `master` (via branche `feature/c2-affinity-diversity`).

---

## 1. Objectif

Transformer le finding **corrélationnel** C0 (« homogénéité d'affinité →
village ») en **mécanisme causal**, par manipulation directe d'une seule
variable : le **nombre d'affinités initiales** des fondateurs.

Chaîne testée (révélée par C0) :
```
diversité d'affinité initiale → diversité de biomes occupés → mobilité / village
```

## 2. Manipulation (une seule variable)

`n_initial_affinities ∈ {1, 2, 4}`. Lever : l'assignation round-robin des
fondateurs passe de `agent_id % 4` à `agent_id % n_initial_affinities`. Le
placement in-affinity (chaque fondateur dans un tile de son biome) est **inchangé**.

| Condition | Affinités fondatrices | Biomes occupés au départ | Prédiction |
|---|---|---|---|
| k=1 | {0} ×20 | 1 | `mobility_score` ↑ (village) |
| k=2 | {0,1} ×10 | 2 | intermédiaire |
| k=4 | {0,1,2,3} ×5 | 4 | `mobility_score` ↓ (mobile) |

## 3. Design apparié (point fort)

Les **mêmes 10 seeds** sont passés dans les 3 conditions. Pour un seed donné, le
`biome_map` (Voronoi, déterministe par seed) est **identique** entre conditions —
**seule la diversité d'affinité initiale change**. Comparaison intra-seed.

**Tenu constant** : biome_map (seed), n_agents=20, max_pop=60, vocalize_cost,
régime `coordination_collective`, ticks (16000), tout le reste.

**Accepté & documenté (PAS un confond à retirer)** : k plus petit → moins de
biomes occupés au départ. C'est **la chaîne causale testée**, pas un artefact.

## 4. Garde-fous d'interprétation (CRITIQUES)

1. **k=1 n'est PAS « meilleur ».** Si k=1 augmente `village_basin` mais **dégrade
   fortement** `gather_successes` / `n_alive` (ou augmente `extinction`), c'est un
   **village écologiquement pauvre**, pas un succès. Le verdict doit lire mobilité
   ET survie ensemble.
2. **Résultat idéal** :
   ```
   k=1 → mobility_score ↑  SANS extinction ↑ massive
   k=4 → mobility_score ↓  avec survie comparable
   ```
   Si une condition casse la survie, l'effet mobilité est inininterprétable seul.

## 5. Plomberie (additive, rétro-compatible)

| Fichier | Changement |
|---|---|
| `aetherlife/world/biomes.py` (`BiomeConfig`) | `+ n_initial_affinities: int = 4` + validation `1 ≤ k ≤ 4` |
| `aetherlife/world/seasonal_grid.py` (`reset`) | `a.agent_id % 4` → `a.agent_id % bcfg.n_initial_affinities` |
| `scripts/overnight_v8b1.py` (`build_env`) | `+ param n_initial_affinities=4` → injecté dans `BiomeConfig` du régime coordination |
| `scripts/overnight_v8b1.py` (`run_overnight`) | `+ param n_initial_affinities` → `build_env` + écrit dans `report["config"]["n_initial_affinities"]` (traçabilité) |
| `scripts/overnight_v8b1.py` (`__main__` argparse) | `+ --n-initial-affinities` (défaut 4) |
| `scripts/run_c2_affinity.ps1` | runner : 10 seeds × {1,2,4} → `results/c2_aff{k}/seed{s}/`, idempotent |
| `scripts/aggregate_c2.py` | agrégation + **deltas intra-seed** (cf. §7) |

**Défaut `n_initial_affinities=4` partout** → comportement actuel inchangé
(non-régression garantie : 5/5/5/5).

## 6. Métriques

- **Primaires** : `mobility_score` (officiel tiers, déjà dans le report via
  `spatial_mobility_v8c3`), `village_basin`. Attendu k=1 > k=2 > k=4.
- **Garde-fous survie** : `n_alive`, `gather_successes_total`, `extinction`
  (n_alive==0).
- **Vérification du mécanisme** : `aff_conc_final` (concentration d'affinité
  finale), `occ_biome_conc` (occupation mono-biome) — doivent monter quand k baisse.

## 7. `aggregate_c2.py` — sortie

Le design étant **apparié**, la sortie doit montrer les **deltas intra-seed**, pas
seulement les moyennes par condition :

```
seed | mobility_k1 | mobility_k2 | mobility_k4 | Δ(k1−k4) | alive_k1 | alive_k4 | gather_k1 | gather_k4
  1  |    0.91     |    0.74     |    0.40     |  +0.51   |   60     |   58     |   120     |   95
 ...
--- moyennes par condition : mobility_score, village_basin %, n_alive, gather, extinction ---
--- Δ(k1−k4) moyen + nb seeds où k1>k4 (test du signe apparié) ---
```

Verdict causal : si `mobility_k1 > mobility_k4` sur la **majorité des seeds
appariés** ET survie comparable → l'homogénéité d'affinité **cause** la sédentarité.

## 8. Tests (TDD, sans GPU)

| Test | Vérifie |
|---|---|
| `BiomeConfig(n_initial_affinities=0)` / `=5` lève `ValueError` | validation bornes |
| `BiomeConfig()` défaut → `n_initial_affinities == 4` | rétro-compat |
| `reset()` k=1 → toutes affinités ∈ {0} | manipulation k=1 |
| `reset()` k=2 → affinités == {0,1}, ~10 chacune | manipulation k=2 |
| `reset()` k=4 → {0,1,2,3} 5/5/5/5 | **non-régression** |
| `run_overnight(n_initial_affinities=k)` → `report["config"]["n_initial_affinities"]==k` | traçabilité |

Le gros est testable sans GPU (config + reset). Les runs eux-mêmes sont le batch.

## 9. Hors scope

- Options « position fixe » et « biome uniforme » (cf. design) = contrôles
  ultérieurs, pas ce premier test.
- C1 food (`food_density_by_biome`) : seulement si C2 ambigu.
- Effet de QUELLE affinité unique pour k=1 (affinité 0 par défaut) : contrôle ultérieur.

## 10. Livrables

- `BiomeConfig.n_initial_affinities` + validation
- `reset()` modifié + tests reset/non-régression
- `build_env` / `run_overnight` / CLI plumbing + test traçabilité
- `scripts/run_c2_affinity.ps1` (runner idempotent)
- `scripts/aggregate_c2.py` (agrégation + deltas intra-seed)
- Finding C2 après batch (mécanisme causal confirmé ou réfuté)
