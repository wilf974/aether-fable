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

## 6quinquies. C0 — résolution corrélationnelle du driver (affinité↔biome, n=20, 2026-05-31)

Test **gratuit, sans GPU, sans recorder v2** : le `biome_map` (Voronoi, statique,
déterministe par seed) se **régénère** via `build_env(seed)` (pur numpy, la
non-déterminisme CUDA n'affecte que le cerveau). Croisé avec les positions+affinité
déjà dans les events.jsonl des 20 seeds.

| Variable | corr avec mobility_score | Lecture |
|---|---|---|
| `aff_biome_match` (agent dans son biome d'affinité) | **−0.087** | **PLAT** : ~0.75 pour TOUS |
| `occ_biome_conc` (occupation concentrée sur 1 biome) | +0.383 | 1 biome → village |
| `aff_conc` (concentration d'affinité) | +0.468 | homogène → village |
| `domAff == domBiome` | +0.105 | 19/20 True (peu discriminant) |

**Résultat clé** : `aff_biome_match` est élevé (~0.75) et **constant** (corr ≈ 0)
— village comme mobile, **tout le monde suit son biome d'affinité** (le gradient
in-affinity food ×1.3 les y attire). Donc « être bien aligné sur son biome »
**n'est PAS** le discriminateur.

**Le discriminateur est l'HOMOGÉNÉITÉ de la population** (`aff_conc` +0.47, son
expression spatiale `occ_biome_conc` +0.38) :

```
mono-affinité  → mono-biome occupé   → VILLAGE   (seed31 : aff_conc 0.99, occ 0.84)
multi-affinité → biomes concurrents  → MOBILITÉ  (seed46 : aff_conc 0.42, occ 0.37)
```

Chaîne causale (corrélationnelle) cohérente avec §6quater :
```
creux profond / homogénéité initiale → monoculture d'affinité → 1 seul biome → VILLAGE
```

> **Reformulation profonde** : la question initiale « la food bouge-t-elle ? »
> s'est transformée en **« la structure sociale héritée (homogénéité d'affinité)
> détermine la géographie collective »**. Mobiles = populations à affinités/biomes
> concurrents (CONFIRMÉ). Villages = mono-affinité (pas « meilleurs dans leur biome »).

**Réserves** : corrélationnel, n=20, **pas encore causal**. `aff_conc`, `creux` et
`occ_biome_conc` sont collinéaires (creux profond *cause* la monoculture, qui
*cause* l'occupation mono-biome). C0 résout le driver au niveau **observationnel** ;
le test causal reste à faire (§7).

**C1 (food_density_by_biome) abandonné comme priorité** : confirmerait surtout une
mécanique déjà codée dans la config (in-affinity ×1.3), sans répondre au « qui
village vs mobile » (= l'homogénéité, pas l'alignement).

## 7. Driver de la mobilité — bornage (résultat négatif de qualité, n=20, 2026-06-01)

> **Mise à jour majeure du statut de C0.** Le candidat C0 (homogénéité d'affinité)
> a été testé causalement (**C2**, finding `2026-06-01-...-c2-affinity-diversity-causal.md`)
> puis temporellement (**C3**, §7 du même finding) : il est **disqualifié comme
> cause de la mobilité**. La monoculture et le village co-émergent d'un goulot
> démographique commun ; le seul effet causal robuste est *diversité → survie*.
> Le driver de la mobilité (village vs mobile **à survie égale**) restait ouvert.
> Cette section **borne ce qu'on peut en expliquer** avec les données actuelles.

Chasse au driver sur les 20 clips (events.jsonl + biome_map régénérable, **0 GPU**),
corrélations et corrélations partielles avec `mobility_score`.

### 7.1 Ce qui est expliqué (positif)

```
goulot démographique (creux = min_alive)  →  tendance à la sédentarisation
```
- `corr(creux, mobility) = −0.48` (creux profond → village), **R² ≈ 0.23**.

> Le goulot démographique est le **meilleur prédicteur observé** de la mobilité,
> mais il n'explique qu'**environ un quart de la variance**. C'est la cause commune
> récurrente du programme (déjà derrière survie, monoculture, formation de village).

### 7.2 Ce qui est réfuté (testé et éliminé)

Aucun n'ajoute de pouvoir explicatif au-delà du creux :

| Variable | corr brute | verdict |
|---|---|---|
| diversité survivante (lignées/affinités post-creux) | −0.42 / −0.37 | **proxy collinéaire du creux** (partielle \| creux → −0.12 ; R²+0.01) |
| timing du creux | −0.08 | nul |
| saison au creux | −0.08 | nul |
| timing de monoculture (t_mono) | −0.06 | nul |
| vitesse de reconstruction post-creux | +0.26 | faible, non robuste |
| taille du biome final occupé | −0.04 | nul |
| concentration dans 1 biome (final) | +0.11 | nul |

> **Aucun déterminant écologique simple extrait des trajectoires (positions,
> affinité, lignée, vocalises, spots, biome_map) n'explique la mobilité au-delà du
> goulot.** Exceptions franches au creux : seed14 (creux profond MAIS mobile),
> seed23 (creux faible MAIS village).

### 7.3 Ce qui reste ouvert (3 hypothèses → 3 générations d'outils)

| Hypothèse | Contenu | Besoin instrumental |
|---|---|---|
| **H1 — Écologie cachée** | food, déplétion locale, gradients non capturés dans events.jsonl | **Recorder V2** (food/biome par tick) |
| **H2 — Politique interne** | stratégie nomade vs sédentaire *apprise* par les lignées survivantes, invisible dans la trajectoire | **OBS Viewer 3** (activations / Q-values / mémoire) |
| **H3 — Contingence historique** | les ~77 % résiduels sont intrinsèquement stochastiques (Gould — thème central du programme) | **Davantage de réplications** |

### 7.4 Conclusion

> À l'état actuel des données, la mobilité apparaît comme un phénomène **faiblement
> contraint** : un goulot démographique influence la probabilité de sédentarisation,
> mais la majeure partie de la variance demeure inexpliquée. Déterminer si cette
> variance résulte d'une **écologie non observée** (H1), d'**états internes des
> politiques** (H2), ou d'une **contingence historique irréductible** (H3) constitue
> la prochaine frontière expérimentale du programme.

Cohérence avec le reste du programme : C0/C2/C3 ont montré que les corrélations
simples cachent souvent une cause commune (le goulot) ; P2 a réfuté l'intuition
« coût → étouffement du signal » ; la mobilité montre maintenant qu'une partie du
système pourrait être **réellement contingente**.

### 7.5 Pilote H1 (food-tracking) — fortement affaibli (2026-06-01)

Test ciblé de H1 *avant* tout batch : Recorder V2 (`record_events_v8`, schema v2,
food par super-cellule 8×8) sur **3 seeds représentatifs** — seed25 (VILLAGE 0.90),
seed46 (MIGRATION 0.15), seed40 (dérive 0.61), 16k ticks. Outil : analyse inline.

| seed | food_at_pop early→late | food_at_pop vs moyenne | tracking pop~food |
|---|---|---|---|
| 25 VILLAGE | 0.8 → 0.3 (×0.40) | 0.5 vs 4.2 (**pauvre**) | **−0.67** |
| 46 MIGRATION | 0.9 → 0.2 (×0.25) | 0.5 vs 3.1 (**pauvre**) | **−0.64** |
| 40 dérive | 0.4 → 0.0 (×0.03) | 0.1 vs 5.0 (**pauvre**) | **−0.65** |

**Résultat principal** : les populations occupent **systématiquement des zones
appauvries** en food (0.5 vs moyenne 4.2) et **anti-corrèlent** avec la food
(~−0.65) — dans les **trois modes identiquement**. La food est **en aval** de la
population (les agents dévorent leur patch ; la food s'accumule là où il n'y a
personne), pas son moteur. Ils ne « suivent pas la nourriture ».

**Réfutation** : les profils food de VILLAGE (seed25) et MIGRATION (seed46) sont
**quasi identiques** (0.5 de moyenne, anti-tracking −0.65) malgré des mobilités
**opposées** (0.90 vs 0.15). Si le moteur était `food locale → décision rester/partir
→ mobilité`, village et migration **devraient diverger** sur ces métriques. Ils ne
divergent pas.

**Conséquence — carte d'hypothèses raffinée** :

```
H1a food-tracking simple      ❌ fortement affaibli (ce pilote)
H1b écologie fine             ? restent possibles : gradients saisonniers fins,
                                compétition inter-biomes, pression de densité locale,
                                structure des ressources non capturée par la moyenne
H2  politique interne         ↑  (la différence est dans ce que les lignées ont APPRIS,
                                pas dans ce qu'elles voient)
H3  contingence historique    ↑
```

On n'élimine **pas toute l'écologie** — on élimine sa **version simple et intuitive**
(« ils migrent parce qu'ils suivent la nourriture »). Le parallèle avec
`mono-affinité → village` (driver apparent, réfuté par intervention, cause réelle
ailleurs) est frappant : même écologie, même food, mêmes règles → **politiques
survivantes différentes** est désormais l'explication de tête.

> **Décision** : NE PAS lancer de batch food complet (le pilote a disqualifié H1a à
> faible coût). La frontière se resserre sur **H2 (état latent des politiques →
> OBS Viewer 3)** et **H3 (contingence → réplications)**. Recorder V2 conservé
> (il a servi à réfuter H1a ; resservira pour H1b si jamais relancé).

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
