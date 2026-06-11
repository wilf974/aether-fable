# Pré-enregistration — Réplication confirmatoire N=100 de l'effet « diversité d'affinité → survie »

> **Statut : VERROUILLÉ le 2026-06-11, AVANT collecte des seeds 11–100.**
> Spec machine : `docs/preregistrations/c2-replication-N100.json` (lockée via
> `scripts/prereg.py lock`). Toute analyse ci-dessous est figée. Aucune
> dérogation au plan sans note datée explicite.

## 0. Relation avec le préreg N=30 (2026-06-10)

Ce préreg **remplace** `2026-06-10-c2-replication-N30-prereg.md`. La collecte
N=30 (seeds 11–30) **n'a jamais eu lieu** : au moment du verrouillage du présent
document, seuls les seeds 1–10 (C2 original, exploratoire) existent dans
`results/c2_aff{k}/`. Par ailleurs son JSON (`c2-replication-N30.json`) n'avait
jamais été locké (`locked_at: null`). Aucune donnée nouvelle n'ayant été vue,
passer directement à N=100 ne constitue pas une dérogation post-hoc : hypothèse,
protocole, métrique primaire et critères de décision sont repris **à l'identique** ;
seul l'effectif change (30 → 100 par condition).

## 1. Contexte et motivation

Le résultat le plus robuste du programme V8-C3 est l'effet **causal protecteur
de la diversité d'affinité sur la survie** :

| Condition | extinction (C2 original, n=10) |
|---|---|
| `k=1` (mono-affinité) | 60 % (6/10) |
| `k=2` | 30 % (3/10) |
| `k=4` (multi, défaut) | 10 % (1/10) |

Mécanisme établi (finding `2026-06-01-finding-v8c3-diversity-as-ecological-insurance.md`) :
**effet portefeuille / assurance écologique** (Yachi & Loreau 1999). Généralité
confirmée : effet **fonctionnel, pas spatial** (finding 2026-06-08).

**Limite statistique** : n=10 par condition → magnitudes bruitées (IC de Wilson
larges). N=100 par condition donne des IC ~±8 pp autour de 50 % et une puissance
largement suffisante pour le contraste k1 vs k4 attendu.

## 2. Hypothèse pré-enregistrée

**H_replication** : l'effet protecteur monotone de la diversité d'affinité sur la
survie se réplique à n=100 : taux d'extinction **strictement décroissant** en k,
écart `k=1 − k=4` substantiel et statistiquement significatif.

H0 : aucune relation monotone entre k et l'extinction (écarts observés à n=10 =
bruit d'échantillonnage).

## 3. Design (identique au protocole C2 original — aucune dérogation)

- Régime : `coordination_collective`
- Ticks : 16 000
- Levier : `--n-initial-affinities ∈ {1, 2, 4}` (manip chirurgicale, cf. spec C2)
- Device : `cuda`
- Seeds : **1–100** par condition (1–10 = C2 original réutilisé ; 11–100 = nouveaux)
- Design **apparié** par seed (même seed → 3 conditions)
- Dossiers : `results/c2_aff{k}/seed{s}/` (idempotent ; runner durci GPU-wait/retry)
- Lancement : `.\scripts\run_c2_replication.ps1 -End 100` (270 runs nouveaux)

## 4. Plan d'analyse (figé)

**Métrique primaire** : taux d'extinction par condition (`final_state.n_alive == 0`),
agrégé via la couche V2.5 (`aetherlife/analysis/`) et `scripts/aggregate_c2.py`.

**Statistiques rapportées** (toutes pré-spécifiées, sans scipy — `analysis/stats.py`) :
1. Taux d'extinction k=1, k=2, k=4 + **IC de Wilson 95 %** par condition.
2. **Test de tendance** : extinction ~ k (pente < 0 attendue) ; Cochran–Armitage
   en contrôle non paramétrique.
3. **Contraste principal** k=1 vs k=4 : z de différence de proportions, IC 95 %
   de la différence (bootstrap `analysis/stats.py` accepté en variante).
4. **Mécanisme** (secondaire, descriptif) : `crash_async`, réservoirs au goulot,
   profondeur du creux — signature portefeuille à n=100.

**Audit machine** : `python scripts/prereg.py audit docs/preregistrations/c2-replication-N100.json
--runs results/c2-replication-N100` (junctions `k{k}` → `c2_aff{k}`, cf. §7).

## 5. Critère de décision PRÉ-ENREGISTRÉ (identique N30)

| Verdict | Condition (toutes requises pour SUCCÈS) |
|---|---|
| **RÉPLICATION RÉUSSIE** | (a) extinction monotone `k1 > k2 > k4` **ET** (b) contraste k1−k4 significatif (p < 0,01) **ET** (c) magnitude `extinction(k1) − extinction(k4)` ≥ **30 points de %** |
| **RÉPLICATION PARTIELLE** | direction monotone respectée mais magnitude < 30 pp **ou** 0,01 ≤ p < 0,05 |
| **ÉCHEC DE RÉPLICATION** | non-monotone **ou** contraste k1−k4 non significatif (p ≥ 0,05) |

Critères machine figés dans le JSON : `extinct(k1) > 0,40` et `extinct(k4) < 0,20`.

**Garde-fou anti-artefact (hérité C2 §4)** : un k=1 qui meurt n'est PAS lu comme un
« village ». L'extinction est la métrique primaire ; la mobilité n'entre pas dans ce critère.

## 6. Ce qui rendrait le résultat soumissible

n=100 par condition, design apparié, hypothèse/critères/seuils figés avant collecte,
audit machine reproductible → niveau preprint (réplication confirmatoire enregistrée).

## 7. Procédure opérationnelle

```powershell
cd "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"

# 1. Campagne (270 nouveaux runs ; seeds 1-10 skip idempotent ; GPU-wait/retry x3)
.\scripts\run_c2_replication.ps1 -End 100

# 2. Junctions pour l'audit (layout results/<prereg_id>/<label>/seed<s>)
New-Item -ItemType Directory -Force results\c2-replication-N100 | Out-Null
foreach ($k in 1,2,4) {
  $j = "results\c2-replication-N100\k$k"
  if (-not (Test-Path $j)) { New-Item -ItemType Junction -Path $j -Target "results\c2_aff$k" | Out-Null }
}

# 3. Audit contre les critères figés
.\.venv\Scripts\python.exe scripts\prereg.py audit docs\preregistrations\c2-replication-N100.json --runs results\c2-replication-N100
```
