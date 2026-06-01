# Finding — La diversité d'affinité protège la population par effet d'assurance écologique (portfolio effect)

**Date** : 2026-06-01
**Phase** : V8-C3 — résultat causal majeur + mécanisme.
**Status** : **Établi** — effet causal monotone reproductible (intervention appariée)
+ mécanisme identifié (réservoirs asynchrones) + cadre théorique (insurance/portfolio
effect, Yachi & Loreau 1999).
**Régime** : `coordination_collective`, baseline, biomes Voronoi (4 affinités,
bonus in-affinity food ×1.3), CUDA.
**Findings liés** : `2026-06-01-finding-v8c3-c2-affinity-diversity-causal.md`
(intervention + réfutation `mono→village`), `2026-05-30-finding-v8c3-coordination-mobility-modes.md`
(C3, co-émergence). Ce finding **extrait et centre** le résultat le plus robuste de
la série.
**Outils** : `scripts/run_c2_affinity.ps1`, `scripts/aggregate_c2.py`,
`scripts/analyze_c2_mechanism.py`, `scripts/record_events_v8.py --n-initial-affinities`.

---

## TL;DR

> Dans AetherLife, **la diversité d'affinité initiale d'une population détermine
> causalement sa survie** face au goulot démographique de fondation — et le
> mécanisme est l'**effet portefeuille / assurance écologique** : plusieurs
> réservoirs (affinité×biome) aux fluctuations démographiques **désynchronisées**
> amortissent le crash agrégé. Quand un réservoir s'effondre, un autre tient ou
> rebondit ; la population totale ne touche jamais le plancher.
>
> ```
> Intervention :  n_initial_affinities = 1 / 2 / 4   (design apparié, 10 seeds)
> Effet :         extinction = 60 % / 30 % / 10 %     (dose-réponse monotone)
> Mécanisme :     réservoirs affinité×biome asynchrones  (crash_async 0 vs 395)
> Cadre :         insurance / portfolio effect (Yachi & Loreau 1999)
> ```
>
> Un principe écologique connu **émerge** de la simulation sans avoir été codé.

## 1. Contexte (C0 → C2 → C3)

L'observation initiale (C0) corrélait homogénéité d'affinité et formation de
« villages » sédentaires. L'intervention causale (**C2** :
`n_initial_affinities ∈ {1,2,4}`, design apparié 10 seeds) a **réfuté** l'hypothèse
naïve `mono-affinité → village` : forcer la mono-affinité ne produit pas des
villages mais de l'**extinction**. L'analyse temporelle (**C3**) a montré que
monoculture et village **co-émergent** d'un goulot démographique commun, sans
causalité directe dans un sens ou l'autre.

Le sous-produit inattendu de C2 — l'effet sur la **survie** — s'est révélé être le
résultat le plus robuste de toute la série. Ce finding le centre et l'explique.

## 2. Effet causal — dose-réponse monotone (C2, batch 16k, 10 seeds/condition)

| Condition (affinités fondatrices) | extinction /10 | n_alive moyen | gather moyen |
|---|---|---|---|
| k=1 (mono) | **6/10 (60 %)** | 24.2 | 26.1 |
| k=2 | **3/10 (30 %)** | 43.0 | 42.6 |
| k=4 (multi) | **1/10 (10 %)** | 55.6 | 94.7 |

Design **apparié** (même seed → même `biome_map` déterministe ; seule la diversité
d'affinité initiale varie — manip chirurgicale, `reset()` +5/−2 lignes, vérifiée en
revue). **Le seul effet causal monotone observé dans tout le programme V8-C3.**

## 3. Mécanisme — tampon par réservoirs asynchrones (batch dynamique, 8 seeds/cond)

Re-capture de trajectoires fines (`record_events_v8 --n-initial-affinities`, 4000
ticks, record_every 5) à travers le goulot (creux ~ t700), k=1 vs k=4 :

| | min_alive (creux) | affinités @creux | biomes @creux | crash_async |
|---|---|---|---|---|
| k=1 (n=8) | **3.2** (jusqu'à 1) | 1.0 | 1.2 | **0** |
| k=4 (n=8) | **13.6** | 3.0 | 3.1 | **395** |

`crash_async` = écart-type des ticks où chaque affinité atteint son **minimum** de
population. k=1 = 0 (1 seul type). **k=4 = 395, chaque run ≥ 94** → les réservoirs
plongent à des moments **nettement décalés**.

> **k=1 n'a qu'UN réservoir** : tout s'effondre en bloc → survie au fil du rasoir
> (jusqu'à 1 survivant). **k=4 a PLUSIEURS réservoirs désynchronisés** : quand l'un
> plonge, un autre tient → l'agrégat reste loin du plancher (creux 4× moins profond).

### Illustration (seed4 k=4, population par affinité)

```
        aff0  aff1  aff2  aff3   TOT
t200      3     8     7    10    28
t600      0     5     7     6    18   ← creux TOTAL peu profond
t1000     0     5    17     8    30
t2000     0     4    50     6    60
```
aff0 meurt tôt (réservoir perdu) ; aff2 touche son fond à t400 puis **rebondit**
(5 → 53) et porte la population ; aff1/aff3 tiennent. L'agrégat (TOT) est **bien
plus stable que n'importe quel réservoir isolé**.

## 4. Cadre théorique — insurance / portfolio effect

Ce comportement est la signature exacte de l'**hypothèse d'assurance de la
biodiversité** (Yachi & Loreau 1999) / **effet portefeuille** : des sous-populations
aux réponses **décorrélées** stabilisent la fonction agrégée, exactement comme un
portefeuille diversifié réduit la variance du rendement total. La protection ne
vient **pas** de l'étendue spatiale (hypothèse H1 *diversification spatiale pure*
**écartée** : le nombre de biomes occupés ne prédit pas la profondeur du creux,
corr −0.22) mais de la **désynchronisation** des réservoirs.

C'est un principe écologique **non codé explicitement** : il **émerge** de
l'interaction entre biomes Voronoi, affinités héritées (bonus food in-affinity) et
sélection démographique — cohérent avec la vocation du projet (émergence vérifiable).

## 5. Articulation avec le reste du programme

- **Variable maîtresse = le goulot démographique** (le « creux »). Il gouverne la
  survie, la monoculture, la formation de village (C3), et — via le nombre de
  réservoirs qui le traversent — le **mécanisme** de survie (ce finding).
- **Cohérent avec la monoculture finale (C3)** : un réservoir l'emporte **après**
  le goulot (aff2 dans l'exemple), mais la diversité était **indispensable pendant**
  le goulot pour le franchir. *Monoculture = état final ; diversité = condition de
  passage.*

## 6. Limites

- Mécanisme : 8 seeds/condition, 4000 ticks (0 extinction observée sur cette
  fenêtre → **profondeur de creux** comme proxy de la survie ; le gradient
  d'extinction vient du batch 16k, §2).
- `crash_async` structurellement 0 pour k=1 (1 affinité) — le signal réel combine
  profondeur (3.2 vs 13.6) + nombre de réservoirs (1.2 vs 3.1 biomes) + asynchronie
  mesurée en k=4 (chaque run ≥ 94 ; déphasage confirmé visuellement).
- **H2 non isolée** : l'origine *comportementale* des réponses asynchrones (les
  lignées explorent-elles différemment, ce qui décorrèle leurs dynamiques ?) reste
  ouverte — nécessiterait une introspection des politiques (OBS Viewer 3).
- Régime unique (`coordination_collective`, 4 biomes). Généralité à tester sur
  d'autres topologies de biomes / nombres de niches.

## 7. Reproduire

```bash
cd "C:/Users/Wilfred/Documents/IA Inst/AetherLife/aetherlife_pkg"
# effet causal (dose-réponse) — batch 16k
& "scripts\run_c2_affinity.ps1" -Start 1 -End 10
python scripts/aggregate_c2.py results/c2_aff1/seed* results/c2_aff2/seed* results/c2_aff4/seed*
# mécanisme — trajectoires fines k=1 vs k=4
for k in 1 4; do for s in 1 2 3 4 5 6 7 8; do \
  python scripts/record_events_v8.py --seed $s --ticks 4000 --record-every 5 \
    --device cuda --n-initial-affinities $k --out-dir results/c2dyn_aff$k/seed$s; done; done
python scripts/analyze_c2_mechanism.py
```
Données (gitignorées, régénérables) : `results/c2_aff{1,2,4}/`, `results/c2dyn_aff{1,4}/`.
