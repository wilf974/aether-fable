# Finding — C2 : la diversité d'affinité est causalement PROTECTRICE ; « mono-affinité → village » RÉFUTÉ

**Date** : 2026-06-01
**Phase** : V8-C3 — test causal de la chaîne corrélationnelle C0.
**Status** : **Inversion causale** — une corrélation naturelle (C0) ne survit pas
à l'intervention. Réfutation propre + finding causal positif inattendu.
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
