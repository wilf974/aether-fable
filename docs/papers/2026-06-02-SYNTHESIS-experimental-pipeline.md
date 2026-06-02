# AetherLife V8-C3 — Synthèse : le pipeline expérimental et les deux couches

*Figure de synthèse — 2026-06-02. Capstone de la séquence mobilité/survie.
Accompagne le quasi-paper `2026-06-02-affinity-diversity-ecological-insurance-PAPER.md`.*

---

## 1. La chaîne d'instruments (pas une chasse aux corrélations)

Chaque outil a été construit pour **fermer une hypothèse**, et s'appuie sur le
précédent. Ce n'est pas une accumulation de mesures — c'est une élimination
instrumentée.

```
  OBSERVATION                INSTRUMENT                 HYPOTHÈSE → VERDICT
  ───────────                ──────────                 ──────────────────
  émergence visible    →  V8 Replay Viewer        →  rend la mobilité VISIBLE
        │
        ▼
  village vs migration →  mobility_score           →  métrique officielle (tiers),
                          (Historian auto-détect.)     Historian la rapporte seul
        │
        ▼
  corrélation           →  régression n=20          →  C0 : homogénéité d'affinité
  mono-affinité↔village                                 ↔ village (corrélationnel)
        │
        ▼
  INTERVENTION          →  n_initial_affinities       →  C2 : mono forcée → EXTINCTION,
  (paired k=1/2/4)         {1,2,4}                       pas village. mono→village ❌
        │
        ▼
  précédence temporelle →  trajectoires fines         →  C3 : mono & village CO-ÉMERGENT
                                                          (goulot commun). village→mono ❌
        │
        ▼
  driver mobilité ?     →  bornage n=20               →  creux R²=0.23, reste 77% inexpliqué
        │
        ├─ écologie ?   →  Recorder V2 (food)         →  H1 ❌ (food village ≈ mobile)
        │
        └─ politique ?  →  OBS Viewer 3 (Q-values)    →  H2 ❌ (politiques indistinguables)
                                                          │
                                                          ▼
                                                    H3 : CONTINGENCE (par élimination)
```

Au passage, l'intervention C2 a produit — inattendu — le résultat le plus robuste
du programme : l'effet protecteur de la diversité sur la survie.

## 2. Le résultat phare — diversité → survie (effet portefeuille)

```
  n_initial_affinities :     1          2          4
  extinction          :     60%   →    30%   →    10%      (dose-réponse monotone)
                              │                     │
                              ▼                     ▼
  1 réservoir                          ~3 réservoirs affinité×biome
  crash synchronisé                    crashs DÉSYNCHRONISÉS (async 0 vs 395)
  creux → 1 survivant                  creux → 13.6 (loin du plancher)
                              │                     │
                              └────────┬────────────┘
                                       ▼
                       EFFET PORTEFEUILLE / assurance écologique
                       (Yachi & Loreau 1999) — ÉMERGE sans être codé
```

> La diversité n'est pas utile parce qu'elle produit de meilleures stratégies ;
> elle est utile parce qu'elle empêche tous les compartiments de s'effondrer en
> même temps.

## 3. Les deux couches du système

Le programme a révélé deux régimes de causalité **coexistants** — la signature des
systèmes biologiques réels.

```
  ┌─────────────────────────────┐     ┌─────────────────────────────┐
  │  COUCHE 1 — LES LOIS         │     │  COUCHE 2 — L'HISTOIRE       │
  │                             │     │                             │
  │  diversité → survie         │     │  village vs migration       │
  │                             │     │                             │
  │  • reproductible            │     │  • mêmes règles             │
  │  • dose-réponse             │     │  • mêmes politiques         │
  │  • mécanisme compris        │     │  • même écologie            │
  │  • prédictible              │     │  → résultat différent       │
  │                             │     │                             │
  │  = effet portefeuille       │     │  = contingence historique   │
  └─────────────────────────────┘     └─────────────────────────────┘
            ↑                                       ↑
   « Pourquoi certaines              « Pourquoi deux populations survivantes
     populations survivent-elles ? »   prennent-elles ensuite des histoires
                                        différentes ? »
            │                                       │
            └───────────── le GOULOT DÉMOGRAPHIQUE ─┘
                    variable organisatrice commune :
            survie · monoculture · villages · diversité résiduelle · (mobilité)
```

**La diversité fixe la *probabilité de passer* le goulot (loi). Le *chemin* pris
ensuite par les survivants est contingent (histoire).**

## 4. Statut des hypothèses (carte finale)

| Hypothèse | Verdict | Instrument |
|---|---|---|
| mono-affinité → village | ❌ réfuté | C2 intervention |
| village → mono | ❌ réfuté | C3 temporel |
| creux/diversité/saison/biome/timing | éliminés | bornage n=20 |
| H1 mobilité = écologie (food) | ❌ réfuté | Recorder V2 |
| H2 mobilité = politique apprise | ❌ réfuté | OBS Viewer 3 |
| **diversité → survie** | ✅ **établi (causal + mécanisme)** | C2 + trajectoires fines |
| **H3 mobilité = contingence** | ✅ par élimination (à caractériser) | réplications (future work) |

## 5. Future work

> Si l'effet portefeuille répond à « pourquoi certaines populations survivent ? »,
> la **prochaine frontière** est : « pourquoi deux populations survivantes prennent
> des histoires différentes ? » — caractériser la contingence par réplications
> contrôlées du même seed (le même monde produit-il parfois village, parfois
> mobile ?), et tester la généralité du portfolio sur d'autres topologies de biomes.
