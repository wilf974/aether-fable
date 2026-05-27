# Finding — B1 : densité seule, hypothèse monotone réfutée — la relation est en cloche

**Date** : 2026-05-27
**Phase** : V8-C3 phase B (test isolé du levier densité)
**Status** : **Hypothèse "densité dominante linéaire" réfutée**, optimum local proche de `max_pop=60` confirmé
**Régime test** : `coordination_collective` C3a''-soft, 15 seeds × 16k ticks, CUDA RTX 3060

---

## TL;DR

> Hypothèse B1 (post-P1) :
>
> > « La densité est le levier dominant ; augmenter `max_pop` de 60 à 70
> > (en gardant `bonus=100`, le seul levier baseline modifié) devrait
> > permettre de monter le % de seeds "very good" vers 35-40 %. »
>
> **Résultat empirique** : 3/15 = **20.0 %**, mean cl_trend = +3.92.
> **Amélioration nette vs P1** (×3) **mais en-deçà du baseline** (~27 %).
>
> L'hypothèse densité dominante **linéaire** est réfutée. La relation
> `max_pop` ↔ `% very good` est **en cloche** avec un optimum local proche
> de `max_pop=60`. Au-dessus, la coop devient trop facile (coop triviale
> par hasard) → l'apprentissage tardif (`cl_trend Q4-Q1`) reste plat
> même quand des très belles trajectoires existent (seed=1 : +22.09).

---

## 1. Configuration testée

| Paramètre | Baseline | P1 | **B1** |
|---|---|---|---|
| `max_pop` | 60 | 50 | **70** |
| `bonus_energy` | 100 | 150 | **100** |
| `vocalize_cost` | 0.05 | 0.05 | 0.05 |

Une seule variable change vs baseline : `max_pop` 60 → 70. Test causal propre.

---

## 2. Résultats par seed

| seed | cl_trend Q4-Q1 | gather | cascade ≥3 | alive | lin | dom_share | very_good |
|---|---|---|---|---|---|---|---|
| 1 | **+22.09** | 59 | 0 | 72 | 3 | 34.21 % | **YES** |
| 2 | +2.20 | 106 | 0 | 73 | 4 | 36.33 % | |
| 3 | +5.56 | 195 | 15 | 71 | 3 | 31.12 % | |
| 4 | +4.61 | 72 | 0 | 71 | 2 | 35.51 % | |
| 5 | −11.18 | 67 | 6 | 72 | 4 | 39.08 % | |
| 6 | −0.06 | 91 | 3 | 72 | 4 | 35.70 % | |
| 7 | +0.04 | 268 | 51 | 72 | 3 | 29.00 % | |
| **8** | **0.00** | **1** | **0** | **0** | **0** | — | **extinction** |
| 9 | −3.59 | 77 | 7 | 73 | 4 | 33.79 % | |
| 10 | +2.32 | 98 | 0 | 70 | 2 | 33.88 % | |
| 11 | **+10.07** | 141 | 0 | 70 | 2 | 27.74 % | **YES** |
| 12 | +8.81 | 298 | 95 | 73 | 5 | 26.80 % | |
| 13 | +2.04 | 66 | 0 | 70 | 3 | 35.91 % | |
| 14 | +5.74 | 114 | 7 | 72 | 5 | 30.45 % | |
| 15 | **+10.11** | 112 | 3 | 71 | 2 | 28.35 % | **YES** |

**Stats globales** : mean = +3.92, median = +2.32, max = +22.09, min = −11.18.
**Very good count** : **3/15 = 20.0 %**.

---

## 3. Lectures qualitatives

### 3.1 Cartographie de la cloche

| Config | max_pop | very good | mean cl_trend |
|---|---|---|---|
| P1 | **50** | 6.7 % | +2.38 |
| Baseline | **60** | ~27 % | (référence) |
| **B1** | **70** | **20.0 %** | +3.92 |

La relation n'est **pas monotone**. Optimum local proche de 60.

### 3.2 Signature inattendue : plus de gathers, mais moins d'apprentissage tardif

B1 produit **beaucoup plus de gathers absolus** que P1 :
- gather max : **298** (seed=12) vs 119 (P1 seed=14)
- cascade ≥3 max : **95** (seed=12) vs 6 (P1 seed=12)

Pourtant `cl_trend Q4-Q1` reste plat (médiane +2.32). Interprétation :
**à `max_pop=70`, la coop devient triviale par hasard partout dès Q1**,
donc l'apprentissage RL n'a pas de pression à concentrer le clustering
tardivement. Q1 et Q4 ressemblent → trend faible.

C'est **exactement le mécanisme diagnostic V8-C3 baseline** : à haute
densité, la coop triviale masque l'émergence. `max_pop=70` ramène ce
problème, en plus modéré que `max_pop=100` (régime C3a' d'origine).

### 3.3 Quand B1 marche, il marche plus fort

- seed=1 atteint **+22.09**, le record absolu sur tous les panels P1+B1+baseline mesurés
- Les 3 seeds "very good" de B1 (1, 11, 15) ont tous max_pop final ≈ 70-72 → population stable
- Mais la *probabilité* d'y arriver reste sous baseline

Donc `max_pop=70` est dans une zone bi-stable : *quand* le régime
émerge, il est plus prononcé qu'à `max_pop=60` (plus de partenaires
disponibles) ; *mais* la probabilité d'émergence baisse.

### 3.4 seed=8 — pathologie systématique

`seed=8` produit une extinction dans **les 3 panels successifs** (P1, B1,
ainsi que dans l'ancien batch P1 v2 avant fix obs où il avait
miraculeusement terminé en 8 s = extinction immédiate). Ce n'est pas du
bruit aléatoire : `seed=8` × `coordination_collective` × `vcost=0.05`
produit une trajectoire RL pathologique (probable boucle d'auto-mort
précoce). À investiguer séparément, mais n'invalide pas les autres
seeds.

---

## 4. Implications pour la SYNTHESIS

### 4.1 Modèle révisé de la double bifurcation

La SYNTHESIS posait :
- Bifurcation 1 : stochasticité initiale → ~27 % atteignent "very good"
- Bifurcation 2 : coût vocalize suffisant → ~75 % conditionnel
- Probabilité conjointe : ~20 %

Mais la SYNTHESIS supposait implicitement une **monotonie locale** des
leviers (« modulé par `max_pop=60` » suggère que c'est un point de
travail, pas un optimum critique). P1 + B1 montrent que **`max_pop=60`
est un optimum local étroit**, pas un point arbitraire d'une famille
linéaire.

### 4.2 Conséquence pratique

Le "bassin d'émergence" est **plus étroit** que la SYNTHESIS le
suggérait. La double bifurcation pourrait être augmentée encore plus
difficilement que prévu — pas seulement à cause de la stochasticité
mais aussi à cause de la **forme du paysage paramétrique** lui-même.

### 4.3 Mise à jour de la stratégie de pilotage

Pour augmenter le taux "very good" :
- **NE PAS** scaler la densité en dehors de la cloche ~50-70
- Explorer plutôt :
  - le **bonus seul** dans la zone densité optimale (B2 = max_pop=60, bonus=120)
  - le **coût vocalize** dans la zone densité optimale
  - les **paramètres architecturaux** (vision_radius, listen_radius)
  - ou la **diversité forcée de lignées** (P5)

---

## 5. Prochain test — B2 (isolation du bonus)

| Test | Config | Hypothèse |
|---|---|---|
| **B2** | `max_pop=60, bonus=120, vcost=0.05` | Bonus seul amplifie sans casser densité |

Critère :
- Si B2 > 27 % → **le bonus est responsable de l'échec P1** (et accessoirement
  un bon levier en zone optimale)
- Si B2 ≈ 27 % ou baisse → **le bonus élevé rend la coop plus triviale**
  et réduit l'apprentissage tardif → le baseline était déjà sur un
  vrai optimum, on ne peut pas le déplacer trivialement
- Si B2 baisse fortement (~6-15 %) → la **densité 60 + bonus modifié**
  casse aussi, et toute la phase B est négative

---

## 6. Notes méthodologiques

- **Run propre** : 15/15 seeds OK, aucun crash (fix obs `fbfb14e` valide)
- **15 seeds × 13 min CUDA** = 3h15 cumulées
- **PYTHONIOENCODING=utf-8** requis
- **Population finale ≈ 70-73** pour les 14 seeds non-extincts (vs 50-53 en P1) — la population stable est plus grosse, comme attendu
- **alive_rate** : 14/15 = 93 % (vs 14/15 en P1, même taux d'extinction)

---

## 7. Provenance

- **Code** : tag `v0.8.17-alpha` + fix obs `8a005ed` + `fbfb14e`
- **Runs** : `results/v8c3b1/seed{1..15}/`
- **CLI** : `python scripts/overnight_v8b1.py --ticks 16000 --regime coordination_collective --vocalize-cost 0.05 --max-pop-override 70 --bonus-energy-override 100 --seed $s --device cuda`
- **Durée** : 13 min/seed sur RTX 3060
- **Réfute partiellement** : hypothèse user post-P1 "la densité spatiale est le levier dominant"

---

## 8. Conclusion

> B1 (`max_pop=70` seul) améliore P1 (×3) sans récupérer la baseline.
> La relation `max_pop ↔ very good` est **en cloche avec optimum proche
> de 60**, pas linéaire. Au-dessus, la coop devient trop facile et
> l'apprentissage tardif n'est plus sélectionné.
>
> Ce finding affine encore la SYNTHESIS : le bassin d'émergence est plus
> étroit que prévu. Le baseline `max_pop=60, bonus=100` est un **vrai
> optimum local**, pas un point de travail arbitraire.
>
> La prochaine étape (B2) isolera le levier `bonus` dans la zone de
> densité optimale pour boucler la causalité de l'échec P1.
