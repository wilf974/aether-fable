# Pré-enregistration — Réplication confirmatoire N=30 de l'effet « diversité d'affinité → survie »

> **Statut : VERROUILLÉ le 2026-06-10, AVANT collecte des seeds 11–30.**
> Toute analyse ci-dessous est figée. Les résultats iront dans un finding séparé
> qui citera ce document. Aucune dérogation au plan sans note datée explicite.

## 1. Contexte et motivation

Le résultat le plus robuste et le plus publiable du programme V8-C3 est l'effet
**causal protecteur de la diversité d'affinité sur la survie** :

| Condition | extinction (C2 original, n=10) |
|---|---|
| `k=1` (mono-affinité) | 60 % (6/10) |
| `k=2` | 30 % (3/10) |
| `k=4` (multi, défaut) | 10 % (1/10) |

Mécanisme établi (finding `2026-06-01-finding-v8c3-diversity-as-ecological-insurance.md`) :
**effet portefeuille / assurance écologique** (Yachi & Loreau 1999) — sous-populations
à réponses décorrélées désynchronisent les crashs (`crash_async` 0 vs 395), stabilisant
l'agrégat à travers le goulot démographique. Généralité confirmée (finding
`2026-06-08-...-portfolio-functional-not-spatial.md`) : effet **fonctionnel, pas spatial**.

**Limite statistique connue et reconnue** : n=10 par condition. Les *directions* sont
robustes, mais les *magnitudes* sont bruitées (IC de Wilson larges). Pour rendre le
résultat soumissible (preprint), cette réplication porte l'effectif à **n=30 par condition**
sous protocole identique. Les seeds 1–10 (C2 original) sont conservés ; on **ajoute** les
seeds 11–30 (20 nouveaux × 3 k = 60 runs) comme **réplication confirmatoire**.

## 2. Hypothèse pré-enregistrée

**H_replication** : l'effet protecteur monotone de la diversité d'affinité sur la survie
se réplique à n=30. Concrètement : taux d'extinction **strictement décroissant** en k,
avec un écart `k=1 − k=4` substantiel et statistiquement significatif.

Hypothèse nulle H0 : aucune relation monotone entre k et l'extinction (les écarts
observés à n=10 étaient du bruit d'échantillonnage).

## 3. Design (identique au protocole C2 original — aucune dérogation)

- Régime : `coordination_collective`
- Ticks : 16 000
- Levier : `--n-initial-affinities ∈ {1, 2, 4}` (manip chirurgicale : affinité des
  fondateurs `% k` ; rien d'autre ne change — cf. spec C2)
- Device : `cuda`
- Seeds : **1–30** par condition (1–10 = C2 original réutilisé ; 11–30 = nouveaux)
- Design **apparié** par seed (même seed → 3 conditions)
- Dossiers : `results/c2_aff{k}/seed{s}/` (idempotent ; runner durci GPU-wait/retry)

## 4. Plan d'analyse (figé)

**Métrique primaire** : taux d'extinction par condition (`n_alive_final == 0`),
agrégé via `scripts/aggregate_c2.py` (champ `extinction` déjà implémenté).

**Statistiques rapportées** (toutes pré-spécifiées) :
1. Taux d'extinction k=1, k=2, k=4 + **IC de Wilson 95 %** par condition.
2. **Test de tendance** : régression logistique extinction ~ k (pente < 0 attendue) ;
   rapporter pente, p, et test de Cochran–Armitage comme contrôle non paramétrique.
3. **Contraste principal** k=1 vs k=4 : test du χ² / z de différence de proportions, IC 95 %
   de la différence.
4. **Mécanisme** (secondaire, descriptif) : `crash_async`, nombre de réservoirs (biomes/
   affinités survivantes) au goulot, profondeur du creux — pour confirmer la signature
   portefeuille à n=30.

## 5. Critère de décision PRÉ-ENREGISTRÉ

| Verdict | Condition (toutes requises pour SUCCÈS) |
|---|---|
| **RÉPLICATION RÉUSSIE** | (a) extinction monotone `k1 > k2 > k4` **ET** (b) contraste k1−k4 significatif (p < 0,01) **ET** (c) magnitude `extinction(k1) − extinction(k4)` ≥ **30 points de %** |
| **RÉPLICATION PARTIELLE** | direction monotone respectée mais magnitude < 30 pp **ou** 0,01 ≤ p < 0,05 |
| **ÉCHEC DE RÉPLICATION** | non-monotone **ou** contraste k1−k4 non significatif (p ≥ 0,05) |

Justification du seuil 30 pp : l'effet original était de **50 pp** (60 % → 10 %). Exiger
≥ 30 pp tolère une régression à la moyenne réaliste tout en restant un effet fort et
publiable. Le seuil est fixé **avant** de voir les nouvelles données.

**Garde-fou anti-artefact (hérité C2 §4)** : un k=1 qui meurt n'est PAS lu comme un
« village ». L'extinction est la métrique primaire ; la mobilité n'entre pas dans ce critère.

## 6. Ce qui rendrait le résultat soumissible

Au-delà de la réplication, le finding citera : intervention dose-réponse (C2) + mécanisme
réservoirs asynchrones (effet portefeuille) + généralité topologique (fonctionnel pas spatial)
+ cadre Yachi & Loreau 1999. La réplication N=30 ferme la dernière réserve statistique.

**Extension optionnelle (non bloquante, à décider après)** : modèle nul analytique
(variance d'un agrégat de k réservoirs à réponses décorrélées) prédisant la pente
extinction(k) — transformerait « on l'observe » en « on le prédit ».

## 7. Exécution

Runner : `scripts/run_c2_replication.ps1` (durci GPU-wait/retry, idempotent).
À lancer **après** la fin du batch P3/P4 en cours (GPU occupé jusqu'au 2026-06-11).

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\scripts\run_c2_replication.ps1            # seeds 1-30 x k{1,2,4}, skip les 1-10 deja faits
python scripts\aggregate_c2.py results\c2_aff1\seed* results\c2_aff2\seed* results\c2_aff4\seed*
```

## 8. Notes d'implémentation (à traiter au moment de l'analyse, en TDD)

`scripts/aggregate_c2.py` couvre la **métrique primaire** (`extinction_pct` par condition)
mais deux extensions sont nécessaires AVANT le verdict — à écrire en TDD contre les vraies
données, pas à l'avance :

1. **Stats inférentielles manquantes** (§4) : IC de Wilson 95 % par condition, régression
   logistique extinction~k (pente + p), contraste k1−k4 (χ²/z + IC de la différence),
   Cochran–Armitage. Prévoir un `aggregate_c2_stats.py` dédié (scipy/statsmodels) plutôt
   que d'alourdir l'agrégateur descriptif existant.
2. **Bug latent à corriger** : dans `summarize_c2`, l'entrée `by_cond[k]` est gardée par
   `if ms:` (mobility scores non vides). Si une condition s'éteint à **100 %**, il n'y a
   aucun survivant → `mobility_score=None` partout → `ms` vide → **la condition disparaît
   du tableau, perdant son `extinction_pct=100 %`**. À n=10 k=1 avait des survivants (40 %),
   donc invisible ; à n=30 le risque demeure pour k=1. Le calcul d'extinction doit être
   sorti du garde `if ms:` (l'extinction se mesure sur `rows`, indépendante de la mobilité).
