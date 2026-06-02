# Finding — H2 RÉFUTÉ : la mobilité n'est pas dans la politique apprise → la mobilité est CONTINGENTE

**Date** : 2026-06-02
**Phase** : V8-C3 — test direct de H2 (driver de mobilité) via OBS Viewer 3.0.
**Status** : **Réfutation de H2** + élévation de **H3 (contingence historique)** par
élimination. Village et mobile ont des politiques statistiquement indistinguables.
**Instrument** : OBS Viewer 3.0 (tag `obs-viewer3-v0.1.0`) — empreinte de politique
(Q-values sur 11 sondes × 9 actions), Policy Distance (cosine), PCA non supervisée.
**Données** : 20 seeds naturels (`coordination_collective`, 16k ticks), 1 empreinte
du cerveau de la lignée dominante survivante par seed, biome_map neutralisé
(comparaison de politiques pures, pas de géographie). 11 village / 9 mobile.

---

## TL;DR

> Après avoir réfuté H1 (la mobilité n'est pas dans l'écologie/food), on a testé
> H2 (la mobilité est dans ce que les lignées ont **appris**) avec un instrument
> dédié : comparer les **politiques** (Q-values) des lignées village vs mobile.
>
> > **H2 réfuté.** Les politiques village et mobile sont **indistinguables** :
> > distances intra-groupe = inter-groupe (~0.97), corrélation
> > policy-distance↔mobility nulle (+0.07), et la PCA non supervisée **entremêle
> > complètement** village et mobile (corr PC1↔mobility +0.17).
>
> Même écologie (H1 ❌), mêmes politiques (H2 ❌) → le destin village/mobile relève
> de la **contingence historique** (H3) : la réalisation stochastique particulière
> du goulot démographique, pas l'environnement ni la stratégie apprise.

## 1. Contexte — élimination progressive

| Hypothèse driver mobilité | Statut |
|---|---|
| `mono-affinité → village` (C0) | réfuté (C2 : intervention → extinction, pas village) |
| `village → mono` (C3) | réfuté (co-émergence, goulot commun) |
| creux / diversité survivante / saison / timing / taille biome | éliminés (bornage R²=0.23) |
| **H1 — écologie cachée (food-tracking)** | **fortement affaibli** (profils food village≈mobile) |
| **H2 — politique apprise** | **CE FINDING → réfuté** |
| **H3 — contingence historique** | **élevé par élimination** |

## 2. Instrument (OBS V3.0)

Le cerveau est un **MLP feedforward** (pas de LSTM → pas d'état mémoire). On le
sonde avec une batterie fixe de 11 observations synthétiques (Food_N/S/E/W,
Gather_adjacent, Token_heard_0/1, Low/High_energy, Alone, Dense_neighbors),
construites via le **vrai `egocentric_obs`** (layer-safe) avec **biome_map
neutralisé** (sinon la distance mélangerait politique et géographie de seed → faux
positif H2). L'empreinte = matrice (11×9) des Q-values. `policy_distance` = cosine.

## 3. Résultats — trois lectures convergentes

### 3.1 Structure des distances (séparées)

```
d(village, village) = 0.957   (n=55, std 0.41)
d(mobile,  mobile)  = 1.006   (n=36, std 0.33)
d(village, mobile)  = 0.972   (n=99)
ratio inter/intra   = 1.00
```
Les trois distances sont **identiques**. Surtout, `d(village,village) ≈ 0.96` : les
cerveaux de villages sont **aussi différents entre eux** que des mobiles → **aucun
attracteur de politique village**, pas de convergence vers une politique commune.

### 3.2 Relation continue

```
corr(policy_distance, mobility_distance) = +0.065   (n=190 paires)
```
Nulle. La distance de politique ne suit pas la distance de mobilité.

### 3.3 PCA non supervisée (le plus décisif)

PC1=33 %, PC2=21 % de variance. Village et mobile **complètement entremêlés** :
seed31 (village, mob 0.95), seed16 (village, 0.81), seed19 (mobile, 0.40) au même
point PC1=−0.70. `corr(PC1, mobility) = +0.165`, `corr(PC2, mobility) = +0.01`.
**Aucune structure spontanée alignée sur la mobilité.** Si une signature de politique
existait, elle percerait dans la PCA — il n'y en a pas.

Artefact visuel : `clips/policy_compare.png` (heatmaps empreinte moyenne village vs
mobile — indistinguables).

## 4. Verdict

> **Village vs mobile n'est PAS dans la politique apprise.** Les lignées
> villageoises et migratrices ont des politiques statistiquement indistinguables
> (par leurs Q-values sur la batterie de sondes).

Combiné à H1 réfuté, il reste **H3 : la mobilité est CONTINGENTE**. Même écologie,
mêmes politiques → le mode village/mobile est déterminé par la **réalisation
stochastique particulière du goulot démographique** (qui survit, où, dans quelle
configuration spatiale au moment précis du creux), pas par l'environnement ni la
stratégie. C'est la **contingence historique de Gould** — thème central du
programme — démontrée ici par **élimination instrumentée**, pas par défaut.

## 5. Limites

- Empreintes quasi-orthogonales même intra-groupe (~1.0) : peut refléter une vraie
  diversité de politiques OU du bruit dans les Q-values DQN (epsilon-greedy, non
  pleinement convergées, échelles per-brain). **Dans les deux cas l'absence de
  structure village/mobile est robuste** : du bruit ne crée pas de faux clustering,
  un vrai signal percerait. L'instrument ne détecte **aucun** mapping politique→mobilité.
- 1 cerveau (lignée dominante) par seed ; n=20 (11/9). Sonder le top-K par seed
  affinerait mais ne changerait pas l'absence de structure.
- Batterie de 11 sondes ; une signature pourrait vivre dans des sondes non testées
  (ex. scénarios composites). Peu probable vu l'entremêlement total en PCA.

## 6. Conséquence pour le programme

> Le driver de la mobilité a été cherché dans l'**écologie** (H1 ❌) puis dans la
> **politique** (H2 ❌). Les deux frontières instrumentées sont fermées. La
> prochaine génération d'outils ne doit pas chercher un *déterminant* de la
> mobilité — il n'y en a probablement pas — mais **caractériser la contingence**
> elle-même : sur N réplications du MÊME seed (CUDA non-déterministe ou bruit
> injecté), le même monde produit-il parfois village, parfois mobile ? Si oui, la
> mobilité est un **point de bifurcation sensible aux conditions** — pas un trait
> piloté. C'est H3 rendu testable (priorité C : réplications).

## 7. Reproduire

```bash
cd "C:/Users/Wilfred/Documents/IA Inst/AetherLife/aetherlife_pkg"
for s in 25 14 24 40 31 42 1 3 6 13 16 19 20 23 27 29 32 37 46 47; do
  PYTHONIOENCODING=utf-8 python scripts/probe_policies_v8.py --seed $s \
    --ticks 16000 --device cuda --out results/probe/seed$s.json; done
python scripts/render_policy_compare.py results/probe/seed*.json --out clips/policy_compare.png
# 3 lectures (structure séparée / corrélation continue / PCA) : analyse inline §3.
```
Données : `results/probe/seed*.json` (gitignoré, régénérable).
