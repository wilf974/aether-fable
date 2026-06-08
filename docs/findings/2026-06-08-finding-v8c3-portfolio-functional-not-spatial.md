# Finding — Le portfolio effect généralise, mais dépend de la diversité FONCTIONNELLE, pas spatiale

**Date** : 2026-06-08
**Phase** : V8-C3 — test de généralité du résultat phare (effet portefeuille).
**Status** : **Généralisation mécaniste** — le portfolio effect (diversité → survie)
réplique à travers les topologies spatiales ; un mécanisme alternatif (réservoirs
purement spatiaux) est explicitement testé et **rejeté**. Le tampon vient de la
diversité de **types** (réponses décorrélées), pas de la multiplicité de patches.
**Protocole** : grille appariée `k ∈ {1,4}` × `n_seed_points ∈ {4,8,16}` × 8 seeds
(48 runs overnight 16k, CUDA), régime `coordination_collective`. Seule la
granularité spatiale (`n_seed_points`) et la diversité d'affinité (`k`) varient ;
biome_map déterministe par (seed, n). Infra : tag `portfolio-topology-v0.1.0`.
**Findings liés** : `2026-06-01-finding-v8c3-diversity-as-ecological-insurance.md`
(C2 + mécanisme §8), `2026-06-01-finding-v8c3-c2-affinity-diversity-causal.md`.

---

## TL;DR

> C2 a établi que la diversité d'affinité **cause** la survie (extinction 60/30/10 %
> pour k=1/2/4), via un **effet portefeuille** : réservoirs affinité×biome aux
> fluctuations désynchronisées (§8). Question ouverte : ce mécanisme est-il **spatial**
> (plusieurs patches) ou **fonctionnel** (plusieurs types) ? On fragmente l'espace
> (`n_seed_points` 4→16) pour éclater la mono-affinité (k=1) en plusieurs patches.
>
> > **Fragmenter l'espace ne sauve PAS la monoculture** : extinction k=1 =
> > 3/8 → 5/8 → 5/8 (n=4→8→16) — plat-à-montant, jamais la chute qu'aurait prédite
> > l'hypothèse spatiale. Mais **le portfolio effect tient à toutes les topologies**
> > (k=4 < k=1 extinction partout, gap +25/+50/+25 pp).
>
> Conclusion : le portfolio effect **généralise** (pas un artefact de Voronoi-8), mais
> repose sur la **diversité FONCTIONNELLE**. Des patches du même type répondent de la
> même façon → fluctuations **corrélées** → aucun tampon. Seuls des types différents
> donnent des réponses décorrélées.

## 1. Progression logique

| Étape | Résultat |
|---|---|
| C2 (intervention) | diversité d'affinité → survie (causal, dose-réponse) |
| §8 (mécanisme) | effet portefeuille — réservoirs asynchrones |
| **Ce test** | le mécanisme est-il spatial ? → **NON. Fonctionnel.** |

C'est la séquence qui fait passer un mécanisme *plausible* à un mécanisme *crédible* :
on teste explicitement l'explication alternative (spatiale) et on la rejette.

## 2. Résultat principal — fonctionnel, pas spatial

Grille d'extinction (sur 8 seeds/cellule) :

| | n=4 | n=8 | n=16 |
|---|---|---|---|
| **k=1** (mono) extinction | 3/8 (38 %) | 5/8 (62 %) | 5/8 (62 %) |
| **k=4** (multi) extinction | 1/8 (12 %) | 1/8 (12 %) | 3/8 (38 %) |
| gap (k1−k4) | +25 pp | +50 pp | +25 pp |

**Forme de la courbe k=1(n)** : 38 % → 62 % → 62 %. **Montante-puis-plate, jamais
décroissante.** L'hypothèse spatiale (H_spatial) prédisait une **chute** (les patches
dispersés de la mono-affinité formant des réservoirs protecteurs). On observe l'inverse :
fragmenter l'habitat d'un type unique **n'aide pas, voire aggrave**.

**Test de convergence** (décisif) : k=1 @ n=16 = **62 %** vs k=4 @ n=4 = **12 %**. Ils
ne convergent pas → réservoirs spatiaux ≠ réservoirs de types.

**Mécanisme affûté (confirme §8 par un nouvel angle)** :

> Des patches de la **même** affinité répondent **identiquement** à l'environnement →
> leurs creux démographiques sont **corrélés** (synchrones) → aucun amortissement.
> Le portfolio effect exige des réponses **décorrélées**, ce que seule la diversité de
> **types** fournit. C'est exactement le `crash_async` de §8 : l'asynchronie naît de la
> diversité fonctionnelle. Multiplier l'espace d'un seul type ne désynchronise rien.

## 3. Le portfolio effect GÉNÉRALISE

À **chaque** topologie, k=4 a moins d'extinction que k=1 (+25/+50/+25 pp). L'effet
diversité → survie n'est **pas** un artefact du monde Voronoi à 8 seeds : il réplique
sous granularité grossière (n=4), standard (n=8) et fine (n=16). (Réplication du
baseline C2 : k=1 @ n=8 = 62 % ≈ les 60 % de C2.)

> Avant ce test, on pouvait objecter : « l'effet portefeuille n'est peut-être qu'un
> artefact de la géographie Voronoi-8. » Après : l'objection ne tient plus.

## 4. Carte des hypothèses (finale)

| Hypothèse | Verdict |
|---|---|
| Diversité **spatiale** protège (multiplicité de patches) | ❌ réfutée |
| Diversité **fonctionnelle** protège (types/réponses distinctes) | ✅ |
| Portfolio effect spécifique à Voronoi-8 | ❌ réfutée |
| Portfolio effect généralise hors topologie d'origine | ✅ |
| Un réservoir = une zone spatiale | ❌ |
| **Un réservoir = une réponse écologique distincte** | ✅ (contribution conceptuelle la plus durable) |

## 5. Observation secondaire — la fragmentation est un stresseur

Distincte du résultat principal (et plus locale au monde AetherLife) : la
fragmentation élevée (n=16) **dégrade tout le monde**, même k=4 (le condition protégé) :

```
k=4 extinction : 12 % → 12 % → 38 %     (saut à n=16)
k=4 gather_moy : 72  → 102 → 43         (effondrement à n=16)
k=4 alive_moy  : 53  → 54  → 38.5
```

Un patchwork fin (petits patches dispersés) augmente le **coût de coordination** : les
agents peinent à se regrouper pour `gather_collective`. C'est un résultat écologique en
soi — **la fragmentation environnementale réduit la survie indépendamment de la
diversité** — qui pourrait ouvrir une ligne de recherche dédiée. Conséquence
méthodologique : la cellule n=16 mélange « effet de diversité » et « stress de
fragmentation » ; mais le résultat principal (k=1 ne s'améliore PAS avec n) est
indépendant de ce confond — si le spatial aidait, k=1 baisserait, il monte.

## 6. Limites

- 8 seeds/cellule : 3/8 vs 5/8 sur k=1 est modeste statistiquement. La **direction**
  (pas de sauvetage spatial) et la **généralité** (k=4 < k=1 partout) sont robustes ;
  les magnitudes exactes sont bruitées.
- 3 niveaux de `n_seed_points` ; `balanced_seeds=True` (chaque type présent). Un balayage
  plus fin ou `balanced_seeds=False` affinerait.
- Side-finding (fragmentation-stresseur) confond la cellule n=16 — noté, sans impact sur
  le résultat principal.
- Mécanisme confirmatoire : la corrélation des creux entre patches d'un même type n'est
  pas mesurée directement (le résultat survie tranche) ; un re-record ciblé
  (positions par tick, k=1 n=4 vs n=16) la montrerait.

## 7. Reproduire

```bash
cd "C:/Users/Wilfred/Documents/IA Inst/AetherLife/aetherlife_pkg"
& "scripts\run_portfolio_topology.ps1" -Start 1 -End 8   # 48 runs, idempotent
python scripts/aggregate_topology.py results/topo_n4_k1/seed* results/topo_n4_k4/seed* \
    results/topo_n8_k1/seed* results/topo_n8_k4/seed* \
    results/topo_n16_k1/seed* results/topo_n16_k4/seed*
```
Données : `results/topo_n{4,8,16}_k{1,4}/seed{1..8}/` (gitignoré, régénérable).
