# Finding — La « convention linguistique » n'est pas un attracteur sain

**Date** : 2026-05-30
**Phase** : V8-C3 — construction de l'instrument P5 (classificateur d'attracteurs)
**Status** : **Réfutation** — la bimodalité convention/coordination de la SYNTHESIS
ne survit pas à l'opérationnalisation sur 50 seeds. Elle était un **artefact
10-seed + token-identity + seuil `dom_share`**.
**Données** : phase D, 50 seeds, baseline `max_pop=60, bonus=100, vcost=0.05`,
régime `coordination_collective`, 16k ticks (aucun GPU consommé pour ce finding —
analyse pure des rapports existants).
**Tag associé** : *aucun*.

---

## TL;DR

> Avant de tester P5 (« la distribution convention/coordination change si on
> augmente la diversité de lignées »), il fallait un **instrument** capable de
> classer un seed en convention vs coordination. En le construisant sur les 50
> seeds phase D déjà sur disque, l'instrument s'est **auto-réfuté** :
>
> > Sur 50 seeds, **convention saine = 0/50**. Le token dominant ne sépare rien
> > (`{0:14, 1:7, 2:18, 3:11}`). Les features causales censées détecter une
> > « vraie » convention (listener_shift, context_consistency, lineage_conc.)
> > soit suivent la coordination, soit sont saturées. **Un seul attracteur sain
> > existe : la coordination spatiale.**
>
> P5 n'est donc pas testable dans sa forme d'origine : on ne peut pas mesurer
> « convention 20 % → 5 % » s'il n'y a pas 20 % de convention saine au départ.

---

## 1. Origine de la bimodalité (ce qu'on traçait)

`docs/findings/2026-05-25-finding-v8c3-density-as-confounder.md` §3.5, **validé
sur 10 seeds** :

| Mode | Token dominant | dom_share | cl_trend | Seeds (10) |
|---|---|---|---|---|
| Convention sans coordination | **token 0** | 40-42 % | **< 0** | 1024, 200 |
| Coordination sans convention | **token 1/2** | 27-33 % | **+2 à +20** | 42, 256, 2048, 99 |
| Ni l'un ni l'autre | varié | faible | ~0 | 7, 100, 123 |

La SYNTHESIS en a tiré « 2 bassins d'attraction (convention vs coordination) »
et la prédiction P5. La force du signal reposait sur **l'identité du token
dominant** (0 vs 1/2) et un seuil de concentration.

## 2. Reproduction sur 50 seeds — l'identité du token ne sépare rien

Distribution du token dominant sur les 50 seeds phase D :

> **`{token0: 14, token1: 7, token2: 18, token3: 11}`** — étalée sur tous les régimes.

Le mapping « token0 = convention, token1/2 = coordination » est **réfuté**.
Contre-exemple direct : **seed14**, token dominant = **0**, `cl_trend = +17`,
very_good → c'est de la coordination pure sur le token « convention ».
L'identité du token est **du bruit** à 50 seeds.

## 3. Triage des features causales — aucune ne ressuscite la convention

On a cherché un signal « convention » plus riche que `dom_share` (un token qui
*cause* un comportement = un mot), comme suggéré au design de l'instrument :

| Feature | Variance utile ? | Corr. avec coordination (`cl_trend`) | Verdict |
|---|---|---|---|
| `dom_share` (pre-success) | oui | **0.00** (orthogonal ✓) | mais high-dom = groupe incohérent (sains→dom bas, dom haut→mourants) |
| `context_consistency` | oui (0.29–0.89) | −0.03 | meilleur candidat, **mais** les top-cc sont des seeds *coord* |
| `listener_shift` | oui | **+0.34** (contaminé) | tracke la coordination, pas un axe convention |
| `lineage_concentration` | **non** (saturé ~1.0) | ~0 | inutilisable |
| `entropy_ratio` | **non** (saturé ~0.99) | ~0 | inutilisable |

Le point qui tue la convention : les seeds **high-cc + low-coord** (candidats
convention : 49, 9, 12) ont tous un **`dom_share` bas** (0.27–0.33) → « consistants
mais diffus » = fourrageurs stables **sans token unique**, donc PAS de convention.
Le seul seed high-dom + high-cc + low-coord (seed43) est **mourant** (`cl_trend
−15,6`, 26 succès).

**Compte final** : convention saine (`cc≥0.80 ∧ cl_trend<2 ∧ dom≥0.36 ∧ alive ∧
succ≥30`) = **0/50**. Robuste : relâcher les seuils ne donne au mieux que
seed45 (1/50), `cc` médian.

## 4. Carte 2D — `dom_share × cl_trend` (50 seeds)

`V` = very_good · `o` = coordination · `.` = plat/diffus · `x` = mort/dégénéré

```
  28 |        V                               cl_trend
  25 |                              x         
  22 |                                        
  19 |                                        
  15 |            V    V      V               
  12 |          V      V                      
   9 |     oo      o            o   o         
   6 |        o o  o    o  o o oo         o   
   3 |   oo.o oo  .xo       o                 
  +0 | ..  .       .    .     x.             x
  -3 |. . .     ..      .                     
  -7 |                                        
 -10 |                                        
 -13 |           .                            
 -16 |                                  x     
     +----------------------------------------
      0.27      0.33      0.39      0.45    0.50  dom_share
```

Lecture : la coordination (`V`/`o`) occupe le haut, **étalée sur tout l'axe
`dom_share`** (la concentration n'a rien à voir avec elle). La droite (`dom_share
> 0.45`) — là où vivrait une « convention forte » — est **vide ou peuplée de `x`
(mourants)**. Aucun amas « haute concentration + vivant + non-coordonné ».

## 5. Distribution des régimes (50 seeds)

| Régime | n/50 | % |
|---|---|---|
| very_good (coord forte) | 6 | 12 % |
| coordination (`cl≥2`, sain) | 22 | 44 % |
| plat / fourragement diffus | 16 | 32 % |
| mort / dégénéré | 6 | 12 % |
| **convention saine** | **0** | **0 %** |

Coordination totale (very_good inclus) = **28/50 = 56 %**. Le système a **un
seul attracteur émergent sain**.

## 6. Interprétation

> La bimodalité convention/coordination était un **artefact** : petit N (10
> seeds), couplé à une lecture de l'**identité du token dominant** qui ne tient
> pas à 50 seeds, et à un seuil de concentration (`dom_share`) qui isolait
> surtout des seeds **en train de mourir** (un token sur-concentré par effondrement
> de population, pas par convention émergente).

Le système AetherLife V8-C3 n'a pas deux attracteurs linguistiques en compétition.
Il a **un attracteur** — la **coordination spatiale** — plus un non-mode
« fourragement diffus stable » et de l'échec. La « convention » au sens d'un mot
conventionnel partagé et fonctionnel **n'émerge pas comme régime stable**.

Ce n'est pas une faiblesse du projet : c'est un **résultat négatif propre** qui
retire une mauvaise théorie **avant** de brûler du GPU. L'instrument destiné à
mesurer P5 a réfuté la prémisse de P5 dès sa construction.

## 7. Décision

- **Prémisse P5 réfutée** : pas de bassin convention à redistribuer.
- **P5 redéfini** → `P5-coord` : tester si la **diversité de lignées forcée**
  (3+ affinities) augmente le **taux ou la qualité de l'unique attracteur
  coordination** (le seul split réel = coordination vs échec). Voir SYNTHESIS §6
  mise à jour.
- La carte 2D `dom_share × cl_trend` devient la figure de référence du mono-attracteur.

## 8. Limitations

- `dom_share` et `context_consistency` sont des proxys ; une convention pourrait
  exister sous une métrique non encore instrumentée (ex. corrélation token↔action
  causale fine). Mais sur **tout le jeu de features causales disponibles**, le
  résultat est négatif et cohérent — pas un artefact d'un seul seuil.
- 50 seeds, un seul régime (`coordination_collective`, baseline). Un autre régime
  pourrait ouvrir un bassin convention ; non testé.

## 9. Reproduire

```bash
cd "C:/Users/Wilfred/Documents/IA Inst/AetherLife/aetherlife_pkg"
source .venv/Scripts/activate
python - <<'PY'
import sys, glob, statistics as st; sys.path.insert(0,'scripts')
from aggregate_v8c3 import _load_report, _extract_metrics, _verdict_per_seed
# extrait dom_share, cl_trend, context_consistency, regime par seed — cf. finding
PY
```
(analyse pure sur `results/v8c3d/seed*/` — gitignoré, régénérable via le
launcher phase D).
