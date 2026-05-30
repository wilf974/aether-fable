# Finding (préliminaire) — Modes de mobilité de la coordination : village / dérive / migration

**Date** : 2026-05-30
**Phase** : V8-C3 — première exploitation scientifique du V8 Replay Viewer
**Status** : **Découverte préliminaire** (n=4 seeds) — un axe de variation spatial
inédit, invisible aux métriques existantes. Classifieur validé ; driver candidat
non confirmé.
**Données** : 4 very_good seeds (25, 14, 24, 40), régime `coordination_collective`,
baseline, 16k ticks, re-simulés via `record_events_v8.py` (events.jsonl par tick).
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

## 5. Driver candidat (non confirmé, n=4)

**Sévérité du creux démographique** : seed14 (migration) a frôlé l'extinction
(**4 survivants** à t≈650), bien plus bas que les villages (13-14) et la dérive
(24). Hypothèse mécaniste :

> Un creux profond ne laisse que quelques **fondateurs**, possiblement déplacés
> hors de la zone d'origine. La repopulation reconstruit le village là où sont les
> fondateurs → relocalisation. Un creux modéré laisse assez de survivants dispersés
> autour de la zone d'origine pour la reconstruire **sur place** → village.

⚠️ **Non monotone sur 4 seeds** : seed40 a le creux le plus *faible* (24) mais
n'est pas le village le plus net (dérive 0.611). Le driver est **suggestif, pas
établi**. Confondants possibles : timing du creux, saison, position exacte des
survivants.

## 6. Limitations

- **n=4** — classification robuste, driver spéculatif. Il faut 10-20 very_good
  seeds pour (a) mesurer le **taux** de chaque mode et (b) tester la corrélation
  creux↔mobilité.
- Re-runs CUDA non bit-exacts : instances représentatives du régime, pas réplique
  des runs phase D.
- Métrique 8×8 / tiers : un balayage de résolution (bins, fenêtres) renforcerait
  la robustesse.

## 7. Suite

1. **Élargir l'échantillon** : record 10-20 very_good seeds, mesurer le taux
   village/dérive/migration et tester le driver « creux profond → migration ».
2. **Officialiser la métrique** : intégrer `corr_occupation` (+ profondeur de
   creux) dans le pipeline d'agrégation comme dimension de caractérisation.
3. Si la migration se confirme rare-mais-réelle et liée au creux : concevoir un
   régime à **ressource non-stationnaire** pour *induire* la migration et
   l'étudier comme comportement collectif à part entière.

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
