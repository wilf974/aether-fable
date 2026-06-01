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

## 3. Réinterprétation de C0 (hypothèse parcimonieuse, PAS fait établi)

C0 observait `mono-affinité ↔ village` parmi les runs naturels. Le causal réfute
`mono → village`. L'explication **la plus parcimonieuse compatible avec les données** :

```
village → monoculture d'affinité   (et non monoculture → village)
```

Une population qui **réussit à s'installer** (village : se fixe dans un biome,
y prospère) voit naturellement l'affinité de ce biome **gagner** par sélection —
la monoculture est une **conséquence** du succès du village, pas sa cause.

> ⚠️ **Prudence** : le causal a réfuté `mono → village`. Il n'a **pas démontré
> formellement** `village → mono` — il a seulement rendu cette direction beaucoup
> plus crédible (en éliminant la direction inverse). À confirmer par un test dédié
> (ex. mesurer la dérive d'affinité *dans* les villages naturels au cours du run).

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
- Test futur de `village → mono` : suivre la trajectoire d'affinité *à l'intérieur*
  des villages naturels (les 11 villages de la distribution n=20) — la monoculture
  s'installe-t-elle APRÈS la sédentarisation ?

## 6. Reproduire

```bash
cd "C:/Users/Wilfred/Documents/IA Inst/AetherLife/aetherlife_pkg"
& "scripts\run_c2_affinity.ps1" -Start 1 -End 10   # 30 runs, idempotent
python scripts/aggregate_c2.py results/c2_aff1/seed* results/c2_aff2/seed* \
    results/c2_aff4/seed*
```
Reports : `results/c2_aff{1,2,4}/seed{1..10}/` (gitignorés, régénérables).
