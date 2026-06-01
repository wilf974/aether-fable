# Finding — C2 : la diversité d'affinité est causalement PROTECTRICE ; « mono-affinité → village » RÉFUTÉ

**Date** : 2026-06-01
**Phase** : V8-C3 — test causal de la chaîne corrélationnelle C0.
**Status** : **Inversion causale + mécanisme établi** — une corrélation naturelle
(C0) ne survit pas à l'intervention ; l'effet causal positif (diversité → survie)
est mécaniquement expliqué (§8 : tampon par réservoirs asynchrones / effet
portefeuille).
**Protocole** : `n_initial_affinities ∈ {1, 2, 4}`, design **apparié** 10 seeds,
régime `coordination_collective`, baseline, 16k ticks, CUDA. Seule la diversité
d'affinité initiale des fondateurs varie (manip chirurgicale, biome_map identique
par seed). Infra : tag `c2-affinity-causal-v0.1.0`.
**Tag associé** : *aucun*.

---

## TL;DR

> C0 (corrélationnel) montrait : parmi les runs naturels, ceux finissant
> mono-affinité étaient des villages. Hypothèse : **mono-affinité → village**.
> On l'a testée causalement en **forçant** la mono-affinité dès l'init.
>
> > Forcer la mono-affinité ne produit PAS de villages — ça produit de
> > l'**extinction**. Gradient dose-réponse monotone, propre, sur 10 seeds/condition :
> > **extinction k=1 = 60 % · k=2 = 30 % · k=4 = 10 %.**
> > La prédiction mobilité (mono → village) **n'apparaît pas** (chez les survivants,
> > c'est plutôt le multi-affinité qui tend à être plus village).
>
> La flèche causale postulée est réfutée. L'effet réel de la diversité d'affinité
> est sur la **survie**, pas sur la sédentarité.

---

## 1. Résultat principal (fort) — la diversité d'affinité cause la survie

| Condition | extinction /10 | n_alive moyen | gather moyen |
|---|---|---|---|
| k=1 (mono) | **6/10 (60 %)** | 24.2 | 26.1 |
| k=2 | **3/10 (30 %)** | 43.0 | 42.6 |
| k=4 (multi) | **1/10 (10 %)** | 55.6 | 94.7 |

**Réponse dose-effet monotone et propre** : plus la diversité d'affinité initiale
est élevée, plus la population survit (et coopère — gather ×3.6 de k=1 à k=4).
Mécanisme plausible : k=1 force les 20 fondateurs dans **un seul biome** Voronoi
(≈ 1/8 de la carte) → surpopulation locale, ressource insuffisante, effondrement.
La diversité d'affinité **diversifie spatialement le risque** sur plusieurs biomes.

> **La diversité d'affinité est causalement protectrice.** C'est le signal le plus
> robuste du protocole (monotone, n=10/condition, biologiquement cohérent).

## 2. Résultat secondaire (réfutation) — aucun support causal pour mono → village

`mobility_score` apparié (exige survie de k_a ET k_b dans le seed) :

| Comparaison | seeds appariés | favorisent + mono | Δ moyen |
|---|---|---|---|
| k1 vs k4 | 4 (k=1 décimé par extinction) | **1/4** → k1 | −0.46 |
| k2 vs k4 | 7 | **3/7** → k2 | −0.21 |

Le gradient prédit (mono → mobility_score ↑) **n'apparaît pas**. Si tendance il y a,
elle est faible, bruitée, et **inversée** : chez les survivants, le multi-affinité
(k=4) tend à être légèrement plus « village » (Δ négatifs). `mobility_score` moyen
des survivants : k=1 = 0.370 (n=4), k=2 = 0.258 (n=7), k=4 = 0.481 (n=9) — non
monotone, dominé par le bruit de petit-n-survivant.

**Garde-fou §4 (spec) appliqué** : on n'a PAS lu la mobilité de k=1 comme
« villages ». k=1 est 60 % mort ; ses rares survivants ne sont pas plus village.
Sans ce garde-fou, on aurait pu conclure à tort « mono → village » sur 4 survivants.

## 3. Réinterprétation de C0 — voir §7 (analyse temporelle C3)

C0 observait `mono-affinité ↔ village`. C2 réfute `mono → village`. La première
hypothèse de clôture était `village → mono` (la monoculture serait une conséquence
du succès du village). **L'analyse temporelle §7 (C3) la corrige** : ni l'une ni
l'autre direction séquentielle ne tient — les deux **co-émergent** suite à un
goulot démographique commun. Lire §7.

## 4. Validité de la manip

- **Design apparié propre** : même seed → même `biome_map` déterministe ; seule la
  diversité d'affinité initiale change (manip `reset()` +5/−2 lignes, vérifiée en
  revue : aucune autre logique, RNG/positions/food/n_agents inchangés).
- **Confond assumé, devenu le finding** : k plus petit → moins de biomes occupés.
  Ce n'était pas un artefact à retirer mais **la chaîne causale testée** — et c'est
  précisément elle qui produit l'extinction, pas un village.
- **Limite** : k=1 est si létal (60 %) que la comparaison de mobilité k1-vs-k4 ne
  porte que sur 4 survivants appariés. Le résultat *survie* est solide (n=10) ; le
  résultat *mobilité* est surtout une **non-confirmation** (pas une réfutation forte
  d'un effet qui pourrait exister à survie égale).

## 5. Décision / suite

- **C0 corrigé** : « homogénéité d'affinité → village » n'est PAS causal. La
  monoculture est vraisemblablement un **effet** du village, pas sa cause.
- **Nouveau finding causal** : diversité d'affinité → survie (effet protecteur).
- **Driver de la mobilité : toujours ouvert.** Le candidat C0 (homogénéité) est
  disqualifié comme cause. Reste à expliquer ce qui rend un village sédentaire vs
  mobile *à survie égale*.
- Test `village → mono` : **fait, voir §7** (a corrigé l'hypothèse de clôture).

## 6. Reproduire

```bash
cd "C:/Users/Wilfred/Documents/IA Inst/AetherLife/aetherlife_pkg"
& "scripts\run_c2_affinity.ps1" -Start 1 -End 10   # 30 runs, idempotent
python scripts/aggregate_c2.py results/c2_aff1/seed* results/c2_aff2/seed* \
    results/c2_aff4/seed*
```
Reports : `results/c2_aff{1,2,4}/seed{1..10}/` (gitignorés, régénérables).

---

## 7. Clôture (C3) — analyse temporelle : co-émergence, pas séquence

Test de précédence sur les 20 clips naturels (events.jsonl, positions+affinité par
tick, **zéro GPU**). Pour chaque seed survivant : `t_mono` (1ʳᵉ fenêtre/10 où
`aff_conc ≥ 0.8 × final`) vs `t_settle` (1ʳᵉ fenêtre dont l'occupation corrèle
≥ 0.8 avec le tiers final).

| | t_settle moy | t_mono moy | mono AVANT settle |
|---|---|---|---|
| VILLAGE (n=11) | **1.2** | 1.5 | **1/11** |
| mobile (n=9) | 3.7 | 2.4 | 6/9 |

### Progression des trois expériences

```
C0 (corrélation)        : mono-affinité ↔ village
        ↓ intervention causale
C2                      : mono-affinité forcée → EXTINCTION, pas village
        ↓ analyse temporelle
C3                      : monoculture et village CO-ÉMERGENT (goulot commun)
```
Chaque étape élimine une interprétation plus naïve.

### Réfutation 1 — la monoculture n'est pas SUFFISANTE pour un village
Les populations **mobiles monoculturisent aussi** (t_mono mobile = 2.4, précoce),
et 6/9 d'entre elles monoculturisent **avant** de se fixer (sans jamais devenir
villages). Donc la monoculture ne cause pas le village.

### Réfutation 2 — le village n'engendre pas ENSUITE la monoculture
Chez les villages, `t_settle ≈ t_mono` (1.2 vs 1.5) et la monoculture ne suit la
sédentarisation que dans **1/11** cas. Aucun délai séquentiel : les deux se
verrouillent **en même temps**, très tôt. L'hypothèse `village → mono` (§2-§5
intermédiaire) est donc **également écartée** dans sa forme séquentielle.

### Modèle retenu — cause commune (goulot démographique)

```
goulot démographique (le « creux » — prédicteur C0 régression : −0.48)
        ↓
sélection d'une affinité dominante  →  monoculture précoce  (conséquence FRÉQUENTE)
        +
stabilisation spatiale éventuelle   →  village              (conséquence CONDITIONNELLE)
```

> **Tous les villages sont monoculturels, mais toutes les monocultures ne
> deviennent pas des villages.** Ce qui distingue village de mobile, c'est le
> **settling** (t_settle 1.2 vs 3.7), pas la monoculture — et le settling n'est
> pas causé par la monoculture (les mobiles monoculturisent sans se fixer).

### Résultat le plus robuste de la série

Ce n'est plus la mobilité (driver toujours ouvert, conséquence secondaire). C'est
l'**effet protecteur causal de la diversité d'affinité sur la survie** :

```
diversité d'affinité ↓  →  risque d'effondrement ↑
k=1 → 60 % extinction   k=2 → 30 %   k=4 → 10 %
```
**Le seul effet causal monotone observé dans toute cette série d'expériences.**

### Conclusion

> Le lien `mono-affinité ↔ village` observé dans les données naturelles (C0) ne
> reflète **ni** une causalité `mono → village` **ni** `village → mono`. Les deux
> phénomènes **co-émergent** suite à un goulot démographique commun. Le seul effet
> causal robuste identifié est l'**effet protecteur de la diversité d'affinité sur
> la survie** de la population.

---

## 8. MÉCANISME de l'effet protecteur — tampon par réservoirs asynchrones (2026-06-01)

C2 établit *que* la diversité protège (dose-réponse). Cette section établit
*comment*. Re-capture trajectoires fines k=1 vs k=4 (`record_events_v8
--n-initial-affinities`, 8 seeds chacun, 4000 ticks, record_every 5) à travers le
goulot. Outil : `scripts/analyze_c2_mechanism.py`.

### 8.1 Résultat

| | extinction | min_alive (creux) | affinités @creux | biomes @creux | crash_async |
|---|---|---|---|---|---|
| k=1 (n=8) | 0/8* | **3.2** (jusqu'à 1) | 1.0 | 1.2 | **0** |
| k=4 (n=8) | 0/8* | **13.6** | 3.0 | 3.1 | **395** |

\* 0 extinction sur 4000 ticks (tirages CUDA) — on mesure la **profondeur du
creux** (cause proximale de l'extinction), pas l'extinction directe. Le gradient
d'extinction vient du batch C2 16k (60 %/30 %/10 %).

`crash_async` = écart-type des ticks où chaque affinité atteint son **minimum** de
population. k=1 = 0 (trivial : 1 seul type). **k=4 = 395, et chaque run ≥ 94** :
les réservoirs touchent leur plancher à des moments **nettement différents**.

### 8.2 Trois hypothèses → verdict

- **H1 (diversification spatiale pure)** : ÉCARTÉ. L'analyse gratuite montrait déjà
  que le nombre de biomes occupés ne prédit pas la profondeur du creux (−0.22).
  Ce n'est pas l'**étendue** spatiale qui protège.
- **H3 (sauvetage démographique par réservoirs asynchrones)** : **CONFIRMÉ**.
- H2 (diversité comportementale) : non isolé ici ; possiblement la cause des
  réponses asynchrones (à creuser via OBS V3).

### 8.3 Le mécanisme — effet portefeuille / assurance écologique

> La diversité protège non pas en **couvrant plus d'espace**, mais en maintenant
> **plusieurs réservoirs (affinité×biome) dont les fluctuations démographiques sont
> DÉSYNCHRONISÉES**. Quand un réservoir s'effondre, un autre tient ou rebondit →
> l'agrégat ne touche jamais le plancher. k=1 n'a **qu'un** réservoir → tout
> s'effondre en bloc → survie au fil du rasoir (min_alive jusqu'à 1).

Illustration (seed4 k=4, population par affinité) :
```
        aff0  aff1  aff2  aff3   TOT
t200      3     8     7    10    28
t600      0     5     7     6    18   ← creux total PEU profond
t1000     0     5    17     8    30
t2000     0     4    50     6    60
```
aff0 meurt tôt (un réservoir perdu) ; aff2 touche son creux à t400 puis **rebondit**
(5→53) et porte la population ; aff1/aff3 tiennent. L'agrégat (TOT) est **bien plus
stable que n'importe quel réservoir isolé** — signature exacte de l'**effet
portefeuille** (hypothèse d'assurance de la biodiversité, Yachi & Loreau 1999) :
des sous-populations à réponses décorrélées stabilisent la fonction agrégée.

### 8.4 Cohérence avec le programme

- Raccorde **C2** (diversité → survie, le *que*) à son *comment* (tampon asynchrone).
- Compatible avec **C3** (monoculture finale) : un réservoir l'emporte **après** le
  goulot (aff2 ici), mais la diversité était **indispensable pendant** le goulot
  pour le franchir. La monoculture est l'état final ; la diversité est la condition
  de **passage**.
- Renforce le rôle central du **goulot démographique** comme variable maîtresse du
  système (survie, monoculture, village — et maintenant le mécanisme de survie).

### 8.5 Limites

- 8 seeds/condition, 4000 ticks (0 extinction observée — profondeur de creux comme
  proxy ; gradient d'extinction du batch 16k).
- `crash_async` structurellement 0 pour k=1 (1 affinité) — le signal réel combine
  profondeur (3.2 vs 13.6) + nb de réservoirs (1.2 vs 3.1 biomes) + asynchronie
  visible en k=4. La désynchronisation n'est pas un artefact (chaque run k=4 ≥ 94,
  trajectoires par affinité déphasées confirmées visuellement).
- H2 (origine comportementale des réponses asynchrones) non isolée → OBS V3.
