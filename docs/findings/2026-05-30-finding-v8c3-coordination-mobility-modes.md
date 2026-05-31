# Finding (préliminaire) — Modes de mobilité de la coordination : village / dérive / migration

**Date** : 2026-05-30
**Phase** : V8-C3 — première exploitation scientifique du V8 Replay Viewer
**Status** : **Confirmé n=20** — un axe de variation spatial inédit, invisible aux
métriques existantes. 3 modes à distribution stable (village 55 % / dérive 30 % /
migration 15 %). Classifieur validé. **Driver « creux profond » RÉFUTÉ.**
**Données** : 20 seeds coordonnés (`coordination_collective`, baseline, 16k ticks),
re-simulés via `record_events_v8.py` (events.jsonl par tick). Mise à jour 2026-05-31
(le préliminaire n=4 reste en §5 pour traçabilité).
**Outil** : `scripts/analyze_occupation_mobility.py`.
**Tag associé** : *aucun*.

---

## TL;DR

> Le V8 Replay Viewer (livré le 2026-05-30) a immédiatement révélé un phénomène
> qu'aucune métrique du projet ne capturait : **la coordination spatiale n'est
> pas un régime unique — elle a des modes de mobilité distincts.**
>
> | seed | corr occupation début~fin | mode | creux (min alive) | zone finale |
> |---|---|---|---|---|
> | 24 | 0.963 | **VILLAGE** | 13 | coin haut-gauche (0,0) → 0.80 |
> | 25 | 0.901 | **VILLAGE** | 14 | coin bas-droite (35,35) |
> | 40 | 0.611 | **dérive partielle** | 24 | bord-haut → coin haut-gauche |
> | 14 | 0.387 | **MIGRATION** | **4** | bas-droite → **bas-gauche** (traverse) |
>
> `cl_trend` (seed14 +14.3, seed25 +27.5) ne dit RIEN de cette différence : la
> mobilité spatiale est **décorrélée** des métriques de coordination/langage.

---

## 1. Contexte

L'analyse des 50 seeds (phase D) et tous les findings antérieurs (P1/B1/P2,
convention réfutée) reposaient sur des métriques **agrégées sans dimension
spatiale** : `cl_trend` (clustering moyen), `dom_share`/entropy (tokens),
`gather_successes`. Les rapports ne stockent aucune position par tick. Le viewer,
via `record_events_v8.py`, produit pour la première fois la **trajectoire spatiale
complète** (positions de chaque agent à chaque tick), ce qui ouvre l'analyse de
mobilité.

## 2. Métrique — corrélation d'occupation début~fin

`analyze_occupation_mobility.py` :
- Découpe la grille 40×40 en **8×8 super-cellules**.
- Construit l'histogramme d'occupation (fraction d'agent-ticks par super-cellule)
  sur le **premier tiers** et le **dernier tiers** du run.
- **corr de Pearson** entre les deux histogrammes :

```
corr > 0.8  → VILLAGE   (agrégation sédentaire, zone dense fixe)
corr < 0.5  → MIGRATION (la zone dense se relocalise)
0.5–0.8     → dérive partielle
```

**Validation du choix de métrique** : le centre de masse (com) est **trompeur**
— il est tiré par la dispersion du creux démographique. Ex. seed24 : com
(16,13)→(2,4) suggère un grand déplacement, mais la **zone dense (mode)** est
(0,0) en début (0.33) ET en fin (0.80) — c'est un village qui se concentre. La
corrélation d'occupation capte le **mode** (robuste), le com capte la **moyenne**
(outlier-sensible). On retient l'occupation.

## 3. Trois modes observés

- **VILLAGE** (seed24, seed25) : la population se fixe sur un coin et y reste tout
  le run. seed24 se concentre extrêmement (80 % des agent-ticks dans une seule
  super-cellule en fin).
- **dérive partielle** (seed40) : glissement lent d'une zone étalée vers un coin,
  sans relocalisation franche.
- **MIGRATION** (seed14) : la zone dense **traverse la carte** (bas-droite →
  bas-gauche), confirmée visuellement (frames `clips/inspect/seed14_debut_t1000`
  et `seed14_fin_t15000`).

## 4. Décorrélations — ce que la mobilité n'est PAS

- **Pas la coordination** : tous les seeds sont very_good (`cl_trend > +10`).
  Village et migration coexistent dans le régime « coordonné ».
- **Pas la lignée / le token** : les 4 seeds deviennent des **monocultures**
  (n_lineages 20→1) et gardent un token dominant **stable** sur tout le run
  (pas de switch au moment du basculement spatial). La prise de pouvoir d'une
  lignée est universelle ; la mobilité varie **indépendamment**.
- **Pas une zone favorisée par l'env** : les villages s'installent dans des
  **coins différents** (seed25 bas-droite, seed24 haut-gauche). Donc **effet
  fondateur stochastique**, pas déterminisme spatial de l'environnement.

## 4bis. Taux des modes (n=20)

| Mode | n/20 | % | seeds |
|---|---|---|---|
| VILLAGE | 11 | **55 %** | 24,25,31,42,6,16,20,23,29,32,37 |
| dérive partielle | 6 | **30 %** | 40,1,3,13,27,47 |
| MIGRATION | 3 | **15 %** | 14 (0.39), 19 (0.40), 46 (**0.15**) |

La migration est un **mode minoritaire réel et reproductible (~15 %)**, pas une
anomalie. Distribution continue de `corr_occupation` (0.15 → 0.97).

## 5. Driver « creux profond → migration » — RÉFUTÉ (n=20)

L'hypothèse préliminaire (n=4) : un creux démographique profond relocalise les
fondateurs → migration. Elle reposait entièrement sur seed14 (4 survivants).

**Réfutée à n=20** :
```
creux moyen : MIGRATION = 16.7   VILLAGE = 11.8   (sens INVERSE de l'hypothèse)
```
Les migrateurs ont un creux **moins** profond en moyenne. seed19 et seed46 migrent
avec des creux peu profonds (24, 22) ; seed14 (creux=4) était une **coïncidence**.
**La profondeur du creux ne prédit pas le mode de mobilité.**

> Premier driver candidat éliminé proprement. Le mode de mobilité reste un
> phénomène robuste **sans cause identifiée** — pistes restantes (sur données
> existantes) : trajectoire multi-fenêtre de la zone dense, coin de settling,
> affinité de la lignée dominante, interaction food/biome, phase saisonnière.

### Préliminaire n=4 (archivé pour traçabilité)
La table n=4 (25,14,24,40) suggérait creux MIGRATION=4 vs VILLAGE=13.5 — artefact
de l'outlier seed14, non répliqué.

## 6. Limitations

- **n=20** — taux et réfutation du driver solides ; cause du mode toujours
  **inconnue**.
- Re-runs CUDA non bit-exacts : instances représentatives du régime, pas réplique
  des runs phase D.
- Métrique 8×8 / tiers : un balayage de résolution (bins, fenêtres) renforcerait
  la robustesse. Seuils village/migration (0.8 / 0.5) à valider sur la distribution
  continue observée.

## 6bis. Chasse au driver — 4 hypothèses testées, 4 réfutées (n=20, 2026-05-31)

| Driver candidat | Mesure | Verdict |
|---|---|---|
| Profondeur du creux | creux MIG=16.7 vs VIL=11.8 | ❌ sens inverse |
| Identité d'affinité (lignée dominante) | MIG=[0,1,3] dispersé | ❌ |
| Concentration d'affinité | VIL 88 % vs mobile 71 % | ~ faible, contre-ex. seed14@99 %, seed16(vil)@59 % |
| Position des survivants au creux | d(surv,début) < d(surv,fin) pour TOUS les modes (MIG 3.0 vs 12.7) | ❌ |

**Résultat structurant** : les migrateurs bottleneckent **dans leur zone d'origine**
(survivants proches du début), puis **relocalisent APRÈS repopulation**. La migration
n'est **pas un effet fondateur** — c'est une **relocalisation post-recovery d'une
population mature**. Signature dynamique : **un saut de quadrant précoce-mais-post-creux
puis re-fixation** (seed46 [2,0,0,0,0], seed19 [1,2,2,2,2]), pas un vagabondage continu.

**Conséquence** : la cause la plus parcimonieuse est **environnementale** (épuisement
de la food locale → rester si elle repousse sur place, partir sinon). Or **la grille
de food n'est pas dans events.jsonl**. → prochain pas-outil identifié.

## 6ter. Métrique OFFICIELLE + correction de fenêtre (2026-05-31)

`mobility_score` est désormais une **métrique officielle de l'Historien**
(`detect_mobility` → discovery `coordination_mobility`), source unique de vérité :
`aetherlife/historian/spatial_mobility.py::window_bounds`, importée par overnight
ET par `analyze_occupation_mobility.py`.

**La fenêtre de mesure ENCODE la définition de « mobilité » — résultat en soi.**
En unifiant la métrique on a découvert que le choix initial (fenêtres début/fin =
**10 %**) mesurait la mauvaise chose :

| Fenêtre | seed31 | seed46 | village_basin (n=20) | Ce qu'elle mesure |
|---|---|---|---|---|
| 10 % début vs fin | 0.043 | 0.052 | 1/20 (5 %) | **founding_relocation** (départ→fin, quasi universel) |
| **tiers (officiel)** | **0.946** | 0.150 | **11/20 (55 %)** | **settled_migration** (un village installé migre-t-il ?) |

Le 1er ~10 % des ticks = transitoire de fondation (creux + dispersion depuis les
positions de départ). seed31 **founde** dans une zone, relocalise une fois vers
t≈1600, puis reste fixe : le 10 % le voit « migrateur » (0.043), les tiers le
voient « village » (0.946) — c'est bien un village. **Le 10 % a été abandonné** ;
`mobility_score` officiel = corr d'occupation **1er tiers vs 3e tiers**,
`village_basin = mobility_score ≥ 0.8`. La distribution n=20 (village 55 %, mean
0.746, continuum + attracteur ~0.95) est confirmée sous la métrique officielle.

## 6quater. Driver — leads sous la métrique OFFICIELLE (régression n=20, 2026-05-31)

Régression de `mobility_score` (tiers, officiel) contre les variables déjà
présentes dans events.jsonl, 20 seeds :

| Variable | corr univ. avec mobility_score | Lecture (haut = village) |
|---|---|---|
| **creux** (min_alive) | **−0.477** | creux PROFOND → village |
| **aff_conc** (concentration d'affinité) | **+0.468** | affinité homogène → village |
| er_mean (energy ratio moyen) | +0.268 | — |
| dom_aff | −0.273 | — |
| vocal_mean | +0.193 | — |
| hot_frac | −0.107 | — |
| n_lin_end | 0.000 | aucun signal |

OLS multivarié : R²=0.53 (n=20, 7 préds → **sur-ajusté, indicatif**).

**Deux prédicteurs de tête (~0.47) racontent la même histoire — l'HOMOGÉNÉITÉ
fondatrice** :
- **Creux profond → village** : peu de survivants → effet fondateur fort →
  monoculture spatiale. Creux faible → survivants dispersés → centres concurrents
  → mobilité.
- **Affinité concentrée → village** : tous préfèrent le même biome → entassement
  dans une zone. Affinités mixtes → tiraillement entre biomes → mobilité.

**Point méthodologique majeur** : le 30/05 le creux avait été « réfuté » comme
driver — mais avec la **mauvaise fenêtre (10 %)**. Sous la métrique correcte
(tiers), le creux **prédit** (signe net, signal réel). *La correction de fenêtre
a débloqué l'analyse causale* : mauvais instrument → faux null → bon instrument →
signal. C'est le finding le plus précieux de la séquence.

**Réserves** : signal modéré (r≈0.47, non déterministe — seed14 a un creux profond
mais est mobile) ; creux et aff_conc probablement **collinéaires** (creux profond
*cause* la monoculture d'affinité) ; n=20 ; R² sur-ajusté. Ce sont des **leads**,
pas un driver résolu.

## 7. Suite — chantier C REFORMULÉ (affinité/biome, pas déplétion food brute)

La régression §6quater redirige C : le mécanisme suspecté n'est pas « la food
bouge » mais **« affinité homogène → installation dans une seule zone-biome →
village »**. D'où :

**Question C** : les villages émergent-ils quand l'affinité dominante correspond
à une **zone-biome stable et concentrée** ?

**Hypothèse C** :
```
affinité homogène + creux fondateur profond → monoculture lignée/affinité
    → occupation d'un biome dominant → VILLAGE
affinités mixtes + creux faible → pôles concurrents → MOBILITÉ / relocalisation
```

**Recorder v2 — champs à capturer en priorité** : `biome_grid`, `food_grid`,
`agent.affinity` (déjà = `aff`), `agent.position`, `local_biome_at_agent`,
`food_density_by_biome`, `occupation_by_biome`. Puis tester la corrélation
`mobility_score` ↔ {concentration d'affinité, adéquation affinité↔biome dominant,
stabilité spatiale du biome occupé}.

Sous-pistes :
- Officialiser `mobility_score` dans `metrics.json` (déjà dans discoveries).
- À terme : régime à **ressource non-stationnaire** pour *induire* la migration.

## 8. Reproduire

```bash
cd "C:/Users/Wilfred/Documents/IA Inst/AetherLife/aetherlife_pkg"
source .venv/Scripts/activate
# (re)générer les events d'un seed
python scripts/record_events_v8.py --seed 14 --regime coordination_collective \
    --ticks 16000 --record-every 10 --device cuda --out-dir results/clip_seed14
# classifier la mobilité
python scripts/analyze_occupation_mobility.py results/clip_seed25 \
    results/clip_seed14 results/clip_seed24 results/clip_seed40
```
