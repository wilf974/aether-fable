# Finding — Phase D : le taux « very good » régresse à 14 % sur 50 seeds

**Date** : 2026-05-29
**Phase** : V8-C3 phase D (solidification statistique du baseline sur 50 seeds)
**Status** : **Recalibrage** — la SYNTHESIS surestimait le *taux* d'occurrence
(27 % → 14 %), mais le **mécanisme** (double bifurcation) reste réel et
reproductible
**Config testée** : baseline `max_pop=60, bonus=100, vocalize_cost=0.05`,
régime `coordination_collective`, 16 000 ticks/seed, CUDA (RTX 3060)
**Tag associé** : *aucun* (le critère de robustesse n'est pas atteint — voir §5)

---

## TL;DR

> La phase B (15 seeds) annonçait un régime « very good » à ~27 %. Le
> SYNTHESIS lui-même prévenait (§7.1) qu'avec 15 seeds × 4 strates, la
> strate « very good » ne contient au plus que 4 seeds — *statistique
> fragile*.
>
> La réplication 50 seeds tranche :
>
> > **very_good = 7/50 = 14,0 %** (6/50 = 12 % en excluant un cas
> > dégénéré éteint). Le 27 % ne réplique pas — régression d'un
> > **facteur ~2** vers la vraie valeur.
>
> Ce n'est **pas** une réfutation. La queue droite est nette (histogramme
> bimodal, 7 seeds `cl_trend > +10`, 2 « ultra » `> +20`), le taux
> d'extinction reste maîtrisé (10 %), et la double bifurcation demeure une
> propriété observable du système. C'est le **taux d'occurrence** de la
> Bifurcation 1 qui était surévalué par un petit échantillon.

---

## 1. Protocole

- **Reprise idempotente** : le batch nocturne s'était arrêté après seed42
  (session terminale fermée, pas un hang). `run_phase_d_resume.ps1 -Start 32
  -End 50` a sauté les 41 seeds déjà complets, nettoyé les dossiers vides, et
  rempli les 9 manquants (38, 43-50). Critère de complétude : présence de
  `overnight_v8b1_seed<N>.json`.
- **Agrégation** : `scripts/aggregate_v8c3.py` sur les 50 dossiers →
  `results/v8c3d_aggregate_50seeds.json`.
- **Définition « very good »** (SYNTHESIS §, Bifurcation 1) :
  `clustering_pre_success.trend_q4_minus_q1 > +10`. Vérifié qu'il n'existe
  pas de champ « control » distinct dans le JSON — le « cl_trend ctrl » du
  SYNTHESIS désignait la *condition* de contrôle, pas un autre champ.

---

## 2. Résultat principal

| Quantité | Valeur |
|---|---|
| **very_good** (`cl_trend > +10`) | **7/50 = 14,0 %** |
| very_good hors dégénéré (seed22 éteint) | 6/50 = 12,0 % |
| ultra (`cl_trend > +20`) | 2/50 = 4,0 % (seeds 22, 25) |
| extinctions | 5/50 = 10,0 % (seeds 8, 21, 22, 26, 50) |

**Les 7 seeds « very good » :**

| seed | cl_trend | gather_succ | alive | note |
|---|---|---|---|---|
| 14 | +17,00 | 88 | 62 | sain |
| 22 | +25,50 | 11 | **0** | dégénéré (éteint, trend bruité) |
| 24 | +16,78 | 43 | 61 | sain |
| 25 | +27,50 | 74 | 61 | sain (ultra) |
| 31 | +12,70 | 45 | 61 | sain |
| 40 | +11,91 | 61 | 60 | sain |
| 42 | +14,34 | 49 | 61 | sain |

**Taux par sous-panel de 10** : 0 % · 10 % · 30 % · 20 % · 10 %.
La variance inter-panel est élevée : un panel de 15 seeds tombant sur la
fenêtre 21-40 aurait facilement mesuré ~25 %, ce qui explique
mécaniquement le 27 % de la phase B.

**Histogramme `cl_trend` (bins de 5)** :

```
[-20,-15) #  (1)
[-15,-10) #  (1)
[ -5, +0) #########  (9)
[ +0, +5) ####################  (20)
[ +5,+10) ############  (12)
[+10,+15) ###  (3)
[+15,+20) ##  (2)
[+25,+30) ##  (2)
```

Distribution clairement à queue droite, mode principal en `[0,+5)`,
avec une sous-population détachée `> +10`. La bimodalité de la
Bifurcation 1 est visible.

---

## 3. Stats agrégées (50 seeds)

| Métrique | mean ± std | min | max |
|---|---|---|---|
| n_alive_final | 54,9 ± 18,5 | 0 | 63 |
| n_births_total | 235,6 ± 102,3 | 28 | 562 |
| gather_successes | 90,9 ± 67,6 | 1 | 362 |
| clustering_mean | 9,66 ± 3,58 | 3,00 | 18,72 |
| **clustering_trend** | **+4,24 ± 7,69** | −15,64 | +27,50 |
| token_dominant_share | 0,367 ± 0,108 | 0,272 | 1,000 |
| token_entropy | 1,309 ± 0,193 | 0,000 | 1,383 |
| cascade_ratio | 0,044 ± 0,065 | 0,000 | 0,243 |

Verdict patterns du script d'agrégation :
`C3b_unlocked_proto_coordination_emerging`
(27/50 « clustering_strong_with_50_succ », 29/50 « apprenable »,
45/50 « no_extinction »).

---

## 4. Interprétation

1. **La SYNTHESIS surestimait le taux, pas le mécanisme.** 14 % au lieu de
   27 % : régression vers la moyenne d'un estimateur à petit N, exactement
   le risque annoncé au §7.1. La double bifurcation reste une propriété
   observable et reproductible (queue droite, sous-population `> +10`).

2. **Probabilité cumulée révisée.** Si Bifurcation 1 ≈ 14 % (au lieu de
   27 %) et Bifurcation 2 ≈ 75 % (inchangé), la probabilité conjointe de la
   transition causale complète passe de **~20 % à ~10,5 %**. À recroiser
   par un futur balayage de coût sur ces 50 seeds.

3. **Cas dégénéré seed22.** `cl_trend=+25,5` mais population éteinte et 11
   succès seulement : trend `q4−q1` calculé sur un micro-échantillon, donc
   bruité (même pathologie que seed8). Un futur critère « very good »
   devrait **gater sur `gather_successes ≥ 30` ET `n_alive > 0`** pour
   exclure ces faux positifs (donnerait 6/50 = 12 %).

4. **Extinctions maîtrisées.** 10 % (5/50), bien sous le seuil de
   disqualification de 30 % du script d'agrégation. Le régime n'est pas
   extinction-dominant.

---

## 5. Verdict selon le critère pré-enregistré (finding phase B §)

| Bande | Lecture | Atteint ? |
|---|---|---|
| [22 %, 32 %] | SYNTHESIS robuste, théorie publiable | ❌ |
| [15 %, 22 %] | valide mais surestimée | ❌ (à 1 pt) |
| **[10 %, 15 %]** | **SYNTHESIS surestimait le taux, mécanisme intact** | ✅ **14 %** |
| < 10 % | réexaminer | — |

**→ Bande [10 %, 15 %].** Pas de tag `v0.10.0-alpha` : le jalon de
robustesse statistique visé n'est pas franchi. Le mécanisme tient, le
chiffre publiable doit être corrigé à **« ~14 % (IC large, forte variance
inter-panel) »**, pas 27 %.

---

## 6. Actions de suivi proposées

- **Corriger la SYNTHESIS** : remplacer « ~27 % » par « ~14 % (50 seeds) »
  partout, et noter le 27 % comme artefact de petit échantillon.
- **Durcir le critère very_good** dans `aggregate_v8c3.py` :
  ajouter le gate `gather_successes ≥ 30 ∧ n_alive > 0` pour une métrique
  robuste aux cas dégénérés.
- **(Optionnel) Balayage de coût sur 50 seeds** pour ré-estimer
  Bifurcation 2 et la probabilité conjointe révisée (~10,5 %).
- **Intervalle de confiance** : avec 7/50, IC95 binomial ≈ [6 %, 27 %] —
  large. Une réplication 100+ seeds resserrerait, mais le rapport
  coût/information décroît (≈ 24 min/seed).

---

## Artefacts

- `results/v8c3d/seed{1..50}/overnight_v8b1_seed{N}.json` — 50 runs bruts
- `results/v8c3d_aggregate_50seeds.json` — agrégat
- `scripts/run_phase_d_resume.ps1` — reprise idempotente
- `scripts/aggregate_v8c3.py` — agrégation
