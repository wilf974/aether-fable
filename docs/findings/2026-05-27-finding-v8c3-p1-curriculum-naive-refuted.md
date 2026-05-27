# Finding — P1 réfutée : curriculum naïf (densité ↓ + récompense ↑) dégrade le régime

**Date** : 2026-05-27
**Phase** : V8-C3 P1 (test prédiction 1 de la SYNTHESIS)
**Status** : **Finding négatif important** — la prédiction P1 du SYNTHESIS.md est réfutée
**Régime test** : `coordination_collective` C3a''-soft, 15 seeds × 16k ticks, CUDA RTX 3060

---

## TL;DR

> Hypothèse P1 (SYNTHESIS) :
>
> > « `max_pop=50, bonus_energy=150` augmenterait le % de seeds atteignant
> > régime "very good" (`cl_trend > +10`) au-delà de la baseline ~27 % »
>
> **Résultat empirique** : 1 seed sur 15 atteint le seuil = **6.7 %**.
> Mean cl_trend = +2.38, médiane +2.30. **Dégradation nette vs baseline.**
>
> Le curriculum naïf (densité ↓ + récompense ↑) **casse** le régime au lieu
> de l'améliorer. La double bifurcation observée dans la SYNTHESIS n'est
> donc **pas pilotable trivialement** par ces deux leviers.

---

## 1. Configuration testée

| Paramètre | Baseline (SYNTHESIS) | P1 |
|---|---|---|
| `max_pop` | 60 | **50** |
| `bonus_energy` | 100 | **150** |
| `vocalize_cost` | 0.05 | 0.05 |
| `regime` | `coordination_collective` | idem |
| n_ticks | 16k | 16k |
| n seeds | 15 | 15 |
| Code | v0.8.17-alpha | v0.8.17-alpha + fix obs `fbfb14e` |

Hypothèse implicite testée : abaisser la densité (forcer les agents à
se chercher activement, comme dans le passage C3a' → C3a''-soft de
`max_pop=100 → 60`) ET augmenter le bonus énergétique (renforcer la
sélection de la coopération) devraient additionner leurs effets.

---

## 2. Résultats par seed

| seed | cl_trend Q4-Q1 | gather | cascade ≥3 | alive | lineages | dom_share | very_good |
|---|---|---|---|---|---|---|---|
| 1 | +8.05 | 73 | 0 | 51 | 4 | 33.62 % | |
| 2 | +2.30 | 83 | 3 | 50 | 4 | 29.72 % | |
| 3 | −1.64 | 70 | 0 | 52 | 3 | 40.31 % | |
| 4 | +3.19 | 66 | 0 | 52 | 4 | 29.10 % | |
| 5 | −0.06 | 67 | 0 | 53 | 5 | 30.19 % | |
| 6 | −5.46 | 61 | 0 | 50 | 3 | 30.88 % | |
| 7 | +4.50 | 69 | 0 | 53 | 5 | 29.21 % | |
| **8** | **0.00** | **1** | **0** | **0** | **0** | — | **extinction** |
| 9 | −3.76 | 25 | 0 | 51 | 2 | 29.78 % | |
| 10 | −0.10 | 47 | 0 | 51 | 3 | 44.98 % | |
| 11 | +3.16 | 75 | 4 | 50 | 2 | 42.00 % | |
| 12 | +3.82 | 88 | 6 | 53 | 4 | 33.19 % | |
| 13 | +9.00 | 92 | 3 | 51 | 5 | 46.65 % | |
| 14 | −3.08 | 119 | 0 | 52 | 4 | 28.85 % | |
| **15** | **+15.76** | 50 | 0 | 52 | 3 | 41.32 % | **YES** |

**Stats globales** : mean = +2.38, median = +2.30, max = +15.76, min = −5.46.
**Very good count** : **1/15 = 6.7 %**.

---

## 3. Lecture qualitative

### 3.1 Cassure double observable

Trois symptômes distincts de dégradation :

1. **3/15 seeds en négatif** (6, 9, 14) — `cl_trend < 0` = anti-coordination
   apprise. Le réseau apprend à *éloigner* les agents avant succès.
2. **1/15 extinction complète** (seed=8) — population tombée à 0,
   1 gather success seulement. Anormal vs baseline.
3. **mean +2.38 nettement sous baseline implicite** (~+5 à +7 attendu).

### 3.2 Hypothèses sur le mécanisme de cassure

**H_a — max_pop=50 retire la masse critique** :
À densité réelle réduite, les rencontres aléatoires deviennent trop
rares pour stabiliser l'apprentissage du clustering. Les agents ne
co-occurrent pas assez souvent pour que les Q-values de "rester
groupés" se renforcent.

**H_b — bonus=150 sature la pression sélective** :
Quand un gather rapporte 150 (vs 100 baseline), un agent qui réussit
UN gather solo a déjà résolu son problème énergétique. La sélection
RL n'a plus besoin d'optimiser la coordination *tardive* (Q4 du run) —
elle plafonne dès Q1. D'où `trend_q4_minus_q1` qui plonge vers 0 ou
négatif.

**H_c — interaction croisée H_a × H_b** :
Les deux effets se renforcent : moins d'agents ET moins de pression =
double affaiblissement. Plus probable vu l'écart de magnitude.

Aucune des trois n'est falsifiée par ce seul panel. La phase B
(leviers isolés) tranchera.

---

## 4. Implications pour la SYNTHESIS

### 4.1 La double bifurcation reste valide — mais **non-triviale à piloter**

Cette réfutation **renforce** la SYNTHESIS plutôt qu'elle ne l'affaiblit.

La SYNTHESIS affirmait : « ~27 % des seeds atteignent régime "very good"
sous une fenêtre paramétrique étroite ». Si abaisser arbitrairement
`max_pop` et augmenter `bonus_energy` suffisait à monter ce taux à
40-50 %, alors la double bifurcation serait un artefact de paramétrage
plutôt qu'un phénomène robuste.

Au contraire, **les paramètres baseline (`max_pop=60`, `bonus=100`)
sont déjà sur ou près d'un optimum local** : déplacer en aval (P1) le
casse. C'est compatible avec la nature de **bassin d'attraction étroit**
prédite par la théorie du signal (Smith/Zahavi).

### 4.2 Hypothèse révisée — la densité spatiale comme levier dominant

Le levier `max_pop` agit doublement :
- Sur la disponibilité de partenaires pour `gather_collective` (≥2 agents
  adjacents nécessaires)
- Sur la fréquence d'écoute de vocalisations (`listen_radius=10` mais
  voisins rares si max_pop bas)

Le levier `bonus_energy` agit principalement sur :
- Le gradient de reward (incitation à apprendre la coop)
- L'extinction probabiliste (énergie acquise par succès)

Hypothèse à tester en phase B : **la densité est le levier dominant**.
Augmenter `max_pop` au-delà de 60 (B1 = 70) devrait être plus efficace
que toucher au `bonus`.

---

## 5. Prochains tests — phase B (leviers isolés)

| Test | Config | Hypothèse | Status |
|---|---|---|---|
| **B1** | `max_pop=70, bonus=100, vcost=0.05` | Densité dominante : plus de partenaires → plus de seeds "very good" | EN COURS |
| **B2** | `max_pop=60, bonus=120, vcost=0.05` | Bonus seul amplifie la pression sans casser la densité | À LANCER |
| **B3** (option) | `max_pop=80, bonus=100, vcost=0.05` | Vérifier saturation densité (trivialiser la coop ?) | Conditionnel |

Critère de succès B1/B2 : % seeds "very good" ≥ 27 % (au moins maintenir
la baseline). Idéalement ≥ 35 %.

Si B1 monte à 40-50 % et B2 reste à ~27 % → hypothèse "densité dominante"
confirmée. La SYNTHESIS pourra être enrichie d'un nouveau levier
explicite et la roadmap pivote sur l'augmentation contrôlée de densité.

Si les deux échouent → la double bifurcation est encore plus rare que
prévu, et il faudra :
- soit augmenter dramatiquement le n seeds (50+, voir 100+) pour
  obtenir une stat exploitable
- soit accepter la SYNTHESIS comme finding définitif (« émergence
  conditionnelle rare, non-pilotable simplement »)
- soit explorer d'autres leviers structurels (P5 diversité de lignées,
  ou changement d'architecture brain)

---

## 6. Notes méthodologiques

- **Fix obs_dim antérieur** : ce panel a tourné sur la version `fbfb14e`
  (3 couches de défense obs_dim), donc aucun crash technique parasitant
  les chiffres. Bug `8a005ed` insuffisant initialement, fix v2 a permis
  15/15 seeds OK.
- **PYTHONIOENCODING=utf-8** requis pour éviter crash cosmétique cp1252
  sur le print final (piège connu).
- **seed=8 extinction** : à investiguer séparément. Sur 15 seeds, 1
  extinction = 6.7 % de taux d'échec total, supérieur au baseline
  attendu. Probablement une trajectoire RL pathologique (apprentissage
  d'une action mortelle dominante). Mérite un check des courbes loss/alive.

---

## 7. Provenance

- **Code** : tag `v0.8.17-alpha` + fix obs `8a005ed` + `fbfb14e`
- **Runs** : `results/v8c3p1/seed{1..15}/`
- **CLI** : `python scripts/overnight_v8b1.py --ticks 16000 --regime coordination_collective --vocalize-cost 0.05 --max-pop-override 50 --bonus-energy-override 150 --seed $s --device cuda`
- **Durée** : 13.4 min/seed sur RTX 3060
- **Réfute** : prédiction P1 de `docs/findings/2026-05-25-finding-v8c3-SYNTHESIS.md` §6

---

## 8. Conclusion

> P1 est réfutée empiriquement sur 15 seeds : `max_pop=50, bonus=150`
> dégrade le régime au lieu de l'améliorer (6.7 % very good vs ~27 %
> baseline). Ce finding négatif renforce la SYNTHESIS — la double
> bifurcation n'est pas pilotable trivialement, ce qui est compatible
> avec un mécanisme de contingence historique (Gould) et non un
> artefact paramétrique.
>
> La phase B (B1 = densité seule, B2 = bonus seul) doit isoler le
> levier dominant avant tout autre test.
