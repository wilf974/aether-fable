# Spec — OBS Viewer 3.0 : Policy Fingerprint & Comparison

**Date** : 2026-06-01
**Sous-projet** : OBS — Brain Observatory, **Viewer 3.0** (introspection cerveau).
Premier slice d'OBS V3 (V3.1 activations live, V3.2 timeline d'apprentissage =
différés, hors scope).
**Statut** : design validé, prêt pour plan TDD.
**Branche cible** : `master` (via `feature/obs-v3-policy-fingerprint`).

---

## 1. Objectif — instrument de H2

Tester si la **mobilité** (village vs migration) est produite par des **politiques
apprises différentes** (H2) ou par des **histoires différentes / contingence** (H3).

> Village → Policy A ?  Migration → Policy B ?  ou  Village ≈ Migration → contingence ?

Le pilote H1 (food-tracking) a été réfuté (même écologie, profils food identiques
village/mobile). H2 est la piste de tête mais **invisible** : on observe positions,
lignées, tokens, villages, migrations — jamais la **politique** qui les produit.
V3.0 rend la politique mesurable et comparable.

## 2. Idée centrale — empreinte comportementale (pas les activations)

Pas les poids, pas les couches, pas les activations. L'**empreinte
comportementale** : nourrir le cerveau d'une lignée avec une **batterie fixe de
sondes** (observations synthétiques) et lire ses **Q-values sur les 9 actions**.

Espace d'actions (régime `coordination_collective`) : **9 actions** =
`[MOVE_0, MOVE_1, MOVE_2, MOVE_3, VOC_0, VOC_1, VOC_2, VOC_3, GATHER]`
(4 moves + 4 tokens de vocalize + 1 gather).

Une empreinte = matrice **(n_sondes × 9)** des Q-values. C'est la signature de la
politique d'une lignée.

## 3. Batterie de sondes

Scénarios synthétiques (≈ 11) :

| Sonde | État construit |
|---|---|
| `Food_N` / `Food_S` / `Food_E` / `Food_W` | une food à 1-2 cases dans la direction |
| `Gather_adjacent` | un gather spot adjacent + ≥1 voisin |
| `Token_heard_0` / `Token_heard_1` | un voisin a vocalisé le token 0 / 1 |
| `Low_energy` | énergie ≈ 10 % max |
| `High_energy` | énergie ≈ 90 % max |
| `Alone` | aucun voisin visible |
| `Dense_neighbors` | plusieurs voisins proches |

## 4. Architecture — 3 modules

| Fichier | Responsabilité | GPU |
|---|---|---|
| `aetherlife/viz/policy_probe.py` | **moteur** : définition des sondes, `build_probe_obs()`, `fingerprint(brain, probes)`, `policy_distance()` | non |
| `scripts/probe_policies_v8.py` | **capture** : run un seed, calcule `mobility_score`, sonde le cerveau dominant en fin de run → JSON | oui |
| `scripts/render_policy_compare.py` | charge N JSON → Policy Distance + heatmaps comparatives PNG | non |

### 4.1 `build_probe_obs` — construction LAYER-SAFE (décision critique)

Les sondes sont construites en appelant le **VRAI `egocentric_obs`** sur un
mini-état d'environnement synthétique (placer food/voisin/spot à la position
relative voulue, régler l'énergie, puis appeler la fonction réelle). Cela
**garantit** que le vecteur 505-dim correspond exactement à ce sur quoi le cerveau
s'est entraîné. Construire le vecteur à la main risquerait des erreurs
d'indexation silencieuses (leçon `dom_share`). On instancie un
`SeasonalMultiAgentFoodGrid` minimal (régime coordination), on le met dans l'état
de chaque sonde, on lit `egocentric_obs(env, agent, vision_radius, listener_vocab,
embedding_dim)`.

### 4.2 `fingerprint(brain, probes)`

Pour chaque sonde, forward du `brain.online(obs)` → vecteur de 9 Q-values (sans
gradient, `torch.no_grad`). Retourne la matrice (n_sondes × 9) + les labels.

### 4.3 `policy_distance(fp_a, fp_b)`

Distance **cosine** entre les empreintes aplaties (vecteurs n_sondes×9). 0 =
politiques identiques, 1 = orthogonales. Symétrique.

### 4.4 `probe_policies_v8.py` (capture)

Réutilise `build_env` + `LineageAgent` (comme le recorder, runner taggé intact).
Run un seed N ticks. Pendant le run : accumule l'occupation (réutilise
`spatial_mobility.window_bounds` + `OccupancyAccumulator`) → `mobility_score` à la
fin (auto-étiquetage : le run calcule SA propre mobilité, pas de dépendance aux
vieux clips, pas de mismatch CUDA). À la fin : identifie la **lignée dominante
survivante** (plus grande population), récupère son `LineageBrain`
(`policy.registry`), le sonde avec la batterie → empreinte. Dump JSON :
```json
{"seed": 25, "ticks": 16000, "mobility_score": 0.90, "village_basin": true,
 "dominant_lineage": 5, "probe_labels": [...], "action_labels": [...],
 "fingerprint": [[q0..q8], ...]}
```
Flags : `--seed --ticks --regime --device --out`.

### 4.5 `render_policy_compare.py`

Charge N JSON. Sépare village (`village_basin=true`) / mobile. Calcule :
- **Policy Distance** : moyennes `dist(village,village)`, `dist(mobile,mobile)`
  (intra-groupe), `dist(village,mobile)` (inter-groupe). Verdict :
  inter ≫ intra → **H2** ; inter ≈ intra → **H3**.
- **Heatmaps** : empreinte moyenne village vs empreinte moyenne mobile, côte à
  côte (pygame → PNG, cellule colorée par Q-value normalisée). + heatmap de la
  **différence** village−mobile (où les politiques divergent).
Sortie : `clips/policy_compare.png` + distances en stdout.

## 5. Test décisif

```
intra = moyenne( dist(village_i, village_j), dist(mobile_i, mobile_j) )
inter = moyenne( dist(village_i, mobile_j) )
inter ≫ intra  → H2 (politiques village vs mobile divergent)
inter ≈ intra  → H3 (contingence : mêmes politiques, histoires différentes)
```

## 6. Données / protocole

Re-run de ~20 seeds (les mêmes que la distribution de mobilité : 25,14,24,40,31,
42,1,3,6,13,16,19,20,23,27,29,32,37,46,47) via `probe_policies_v8.py`. Chaque run
auto-étiquette sa mobilité + produit une empreinte. ~20 runs × 16k ticks (~1h GPU,
batch séparé, hors TDD). Puis `render_policy_compare.py` sur les 20 JSON.

## 7. Tests (TDD, sans GPU)

| Test | Vérifie |
|---|---|
| `build_probe_obs("Food_N")` → canal food activé dans la bonne cellule du vecteur | construction layer-safe |
| `build_probe_obs` → dimension == obs_dim attendu (505 en coordination) | conformité |
| `fingerprint(brain, probes)` → forme (n_sondes, 9), finie | moteur |
| `policy_distance(fp, fp) == 0` ; `policy_distance` symétrique ; orthogonal → 1 | métrique |
| `render_policy_compare` sur 2 JSON synthétiques → PNG non vide + distances calculées | viz + verdict |
| smoke `probe_policies_v8 --ticks 60 --device cpu` → JSON valide (fingerprint forme, mobility_score présent) | capture end-to-end |

Le gros est testable sans GPU (moteur + viz). Capture = 1 smoke court.

## 8. Hors scope (V3.1 / V3.2)

- Activations / poids / couches internes → **V3.1** (dashboard live « voir penser »).
- Évolution temporelle de la politique → **V3.2** (archéologie des lignées).
- Probe de TOUTES les lignées (V3.0 = lignée dominante seulement).
- Sondes au-delà de la batterie fixe (~11).

## 9. Livrables

- `aetherlife/viz/policy_probe.py` (moteur : sondes + build_probe_obs + fingerprint + distance)
- `scripts/probe_policies_v8.py` (capture)
- `scripts/render_policy_compare.py` (Policy Distance + heatmaps)
- `tests/test_policy_probe.py` (moteur + distance + viz, sans GPU)
- `tests/test_probe_policies_v8.py` (smoke capture)
- Finding H2 après batch : politiques divergent (H2) ou non (H3).
