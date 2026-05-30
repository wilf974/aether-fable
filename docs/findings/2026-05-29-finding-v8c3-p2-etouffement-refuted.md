# Finding — P2 : l'« étouffement » à coût vocal élevé est RÉFUTÉ

**Date** : 2026-05-29
**Phase** : V8-C3 — prédiction falsifiable P2 (SYNTHESIS §6)
**Status** : **Réfutation** — la prédiction d'inversion/étouffement à `vcost ≥ 0.1`
ne se vérifie pas dans la fenêtre testée. Le coût élevé agit comme **filtre de
sélection**, pas comme suppresseur.
**Config testée** : baseline `max_pop=60, bonus=100`, régime
`coordination_collective`, **ctrl only**, 16 000 ticks/seed, CUDA (RTX 3060).
Seul `vocalize_cost` varie : **0.1** (15 seeds) et **0.2** (15 seeds).
**Runner** : `scripts/run_p2_etouffement.ps1` (commit `779a029`, idempotent).
**Tag associé** : *aucun*.

---

## TL;DR

> P2 prédisait (SYNTHESIS §6) qu'au-dessus de `vcost = 0.1`, l'effet de la
> signalisation coûteuse devait **s'inverser** : émergence linguistique
> *étouffée*, vocalize effondré, `cl_trend` négatif.
>
> Les 30 runs tranchent dans le **sens inverse de la prédiction** :
>
> > À `vcost 0.1`, le système reste au niveau baseline (very_good durci
> > **13,3 %** vs 14 %). À `vcost 0.2`, il **remonte** dans la bande
> > « robuste publiable » : very_good durci **33,3 %** (5/15), `cl_trend`
> > moyen **+5,47** (> baseline +4,24), vocalize **robuste** (−9 %).
>
> Aucun des 5 signaux d'étouffement n'est observé. Le coût vocal élevé,
> dans la fenêtre 0.1–0.2, ne tue pas le signal : il **polarise** le système
> (extinctions en légère hausse) tout en **renforçant le signal chez les
> survivants** — cohérent avec un mécanisme de handicap (Zahavi), pas de
> suppression.

---

## 1. Protocole

- **30 runs** : 15 seeds × {`vcost=0.1`, `vcost=0.2`}, condition **ctrl**
  (pas d'ablation), tous autres paramètres au baseline.
- Lancé 2026-05-29 ~08:47, batch background idempotent, terminé dans la nuit
  (15/15 + 15/15 fichiers `overnight_v8b1_seed*.json` présents).
- ⚠️ Contention GPU pendant une partie du run (2 jobs MW_IA concurrents) —
  ralentissement, pas d'OOM ni de corruption (tous les seeds complets).
- Agrégation : `scripts/aggregate_v8c3.py` avec **critère durci**
  (`very_good = cl_trend > +10 ∧ gather_succ ≥ 30 ∧ n_alive > 0`).
- Agrégats : `results/v8c3p2_v10_aggregate.json`, `results/v8c3p2_v20_aggregate.json`.

---

## 2. Résultats — les 5 signaux d'étouffement

Baseline de référence = phase D 50 seeds (very_good **14 %**, cl_trend moyen
**+4,24**, gather **91**, extinction **10 %**).

| Signal | baseline (50s) | vcost 0.1 | vcost 0.2 | Étouffé ? |
|---|---|---|---|---|
| **very_good durci** | 14 % | 13,3 % (2/15) | **33,3 % (5/15)** | ❌ *monte* à 0.2 |
| **vocalize total** (moy.) | — | 243 190 | 221 498 (−9 %) | ❌ robuste |
| **cl_trend** (moy.) | +4,24 | +4,66 | **+5,47** | ❌ *monte* |
| **gather_successes** (moy.) | 91 | 101,8 | 76,2 (σ=128,5) | ~ mitigé* |
| **extinctions** | 10 % | 13,3 % (2/15) | 20,0 % (3/15) | légère hausse |

\* `gather` à 0.2 est tiré vers le bas par sa variance énorme (un seed à 528,
plusieurs à 0) — signature de **polarisation**, pas d'effondrement médian.

Bande de robustesse (sortie agrégateur) :
- `vcost 0.1` → `mecanisme_reel_taux_surestime [10%,15%]`, proba conjointe ~10 %.
- `vcost 0.2` → **`robuste_publiable [22%,32%]`**, proba conjointe ~25 %.

Métriques secondaires (vs C3a' attendu pour étouffement : entropy↓, dom_share↑) :
- entropy : 1,348 (0.1) / 1,309 (0.2) — légère baisse à 0.2, loin d'un collapse.
- dom_share : 33,8 % (0.1) / 36,8 % (0.2) — spécialisation modeste, pas de
  monopolisation d'un token unique.
- cascade_ratio : 4,4 % / 4,5 % — stable ; 1 seed `cooperation_cascade_attractor`
  apparaît à 0.2.

---

## 3. Interprétation

La prédiction P2 reposait sur une lecture **monotone** de la signalisation
coûteuse : plus le signal coûte cher, moins on signale, jusqu'à extinction du
canal. Les données disent autre chose dans la fenêtre [0.1, 0.2] :

> **Le coût vocal élevé est un filtre de sélection, pas un suppresseur.**
> La phase haut-coût **polarise** le système : davantage d'extinctions (régimes
> qui ne franchissent pas le seuil énergétique meurent), mais **chez les
> survivants le signal reste robuste, voire renforcé**.

C'est exactement la lecture du **handicap de Zahavi** : un signal plus coûteux
est plus honnête et donc plus sélectivement valorisé chez ceux qui peuvent se
le permettre. Le canal ne s'éteint pas — il se **trie**.

Conséquence sur la carte des phases : la transition d'ordre 1 connue à
`vcost ≈ 0.045` (Δcl +0,86 → −7,61) **ne se prolonge pas** en une seconde
dégradation monotone à coût croissant. S'il existe un vrai seuil
d'étouffement, il est **au-delà de 0.2** (non testé ici — voir §5).

---

## 4. Décision

- **P2 réfutée** dans sa forme « inversion à `vcost ≥ 0.1` ».
- Pas de phase 2 d'ablation ciblée : elle ne se justifiait que si un coût
  *tuait* les very_good. Ici les deux coûts les **conservent** (0.1) voire les
  **boostent** (0.2) — les régimes sont vivants, rien à diagnostiquer en
  post-mortem causal.
- **Stop P2.** Prochain cycle : **P5** (diversité de lignées → bassin
  convention vs coordination), levier structurel plus neuf que pousser `vcost`.

---

## 5. Limitations

- Fenêtre étroite : seuls 0.1 et 0.2 testés. La réfutation porte sur
  « étouffement **précoce** » ; un seuil d'étouffement plus haut (0.4–0.8+)
  reste **non exclu**. Choix assumé de ne pas transformer un résultat clair en
  chasse au seuil potentiellement coûteuse.
- 15 seeds/coût → même fragilité statistique que la phase B (cf. SYNTHESIS
  §7.1). Le 33,3 % à 0.2 est dans la bande publiable mais demanderait 50 seeds
  pour être verrouillé comme loi.
- Contention GPU pendant le run : runs complets et valides, mais durées non
  comparables entre seeds.

---

## 6. Reproduire

```bash
cd "C:/Users/Wilfred/Documents/IA Inst/AetherLife/aetherlife_pkg"
source .venv/Scripts/activate
export PYTHONIOENCODING=utf-8
python scripts/aggregate_v8c3.py --runs results/v8c3p2_v10/seed{1..15} --out results/v8c3p2_v10_aggregate.json
python scripts/aggregate_v8c3.py --runs results/v8c3p2_v20/seed{1..15} --out results/v8c3p2_v20_aggregate.json
```
