# V8-C3 — Actions coopératives lourdes : `gather_collective`

> **Statut** : spec post-multi-seed null V8-C2.b''.
> **Date** : 2026-05-25
> **Verrou** : démontrer (ou réfuter) qu'une tâche intrinsèquement
> coopérative sélectionne fonctionnellement la communication.

---

## 0. Position post-V8-C2

V8-C2.b'' multi-seed a montré : malgré food invisible + vision réduite +
écologie durcie, l'effet d'ablation sur la fécondité reste à zéro
(Δ births = +0.41 % ± 0.35 sur 5 seeds × 10k).

Hypothèse : **la coordination reste optionnelle**. Un agent seul peut
survivre par exploration et reproduction asexuée (un seul parent suffit
en V5+).

V8-C3 introduit une **mécanique qui ne peut PAS être résolue seul**.

---

## 1. Le mécanisme `gather_collective`

### 1.1 Design (selon spec user)

```python
@dataclass(frozen=True)
class CooperativeConfig:
    enabled: bool = False
    # Action requise : 2+ agents adjacents (Manhattan ≤ 1)
    min_partners_adjacent: int = 1   # = 2 agents au total
    # Fenêtre temporelle après signal pour réussir
    signal_window_ticks: int = 5
    # Récompense énergétique
    bonus_energy: float = 30.0
    # Tile spéciale : "gather_spot" qui apparaît parfois
    spawn_density: float = 0.02  # 2 % des tiles libres ont un gather_spot
    spawn_decay_ticks: int = 50  # le spot disparaît après N ticks
    # Action ID 8 (au-delà des 4+4 = move+vocalize)
    action_id: int = 8
```

### 1.2 Mécanique au tick

```
Pour chaque agent qui choisit l'action gather_collective (id=8) :
  1. Si l'agent est sur un gather_spot actif :
     a. Compter les voisins adjacents (Manhattan ≤ 1)
     b. Si ≥ min_partners_adjacent voisins :
        → succès collectif : tous les participants reçoivent +bonus_energy
        → le gather_spot disparaît
     c. Sinon : pas de bonus, mais l'agent perd sa turn (no-op)
  2. Si pas sur un gather_spot : no-op
```

### 1.3 Pourquoi cela force la communication

Un agent isolé qui voit un gather_spot mais n'a pas de voisin **ne peut
pas l'exploiter**. Il doit :
- Soit attendre qu'un voisin passe par hasard
- Soit **vocaliser pour attirer**
- Soit migrer ailleurs

Si plusieurs agents ont appris à associer un token X à "viens près de
moi", alors :
- Un agent émet le token quand il voit un gather_spot
- Les auditeurs proches convergent
- La récompense collective renforce ce comportement

→ **Le langage devient fonctionnellement sélectionné** pour la
coordination spatiale.

### 1.4 Hypothèse falsifiable

**H1** : sous régime coordination_collective (V8-C3) + ablation @ t=15k :
- Naissances chutent > 20 %
- Pop mean chute > 15 %
- food/agent ratio chute
- shift KL d'au moins 1 token > 0.10 dans le témoin

**H0** : encore null. Le langage n'est pas vecteur de coordination même
avec tâche coop. Conclusion forte : revoir l'architecture
(soit le RL ne peut pas apprendre la coordination via langage, soit
le design vocalize est incompatible avec la coordination).

---

## 2. Architecture

### 2.1 Nouveau module `aetherlife/world/cooperative.py`

```python
@dataclass(frozen=True)
class CooperativeConfig:
    enabled: bool = False
    min_partners_adjacent: int = 1
    signal_window_ticks: int = 5
    bonus_energy: float = 30.0
    spawn_density: float = 0.02
    spawn_decay_ticks: int = 50


class GatherSpot:
    """Une tile spéciale qui rapporte +bonus_energy si ≥2 agents s'y
    rassemblent dans une fenêtre temporelle."""
    pos: tuple[int, int]
    spawned_tick: int
    decays_at: int
```

### 2.2 Modification `SeasonalMultiAgentFoodGrid`

- Champ `_gather_spots: dict[tuple[int, int], GatherSpot]`
- `_spawn_gather_spots()` : à chaque tick, spawn des nouveaux spots
- `_decay_gather_spots()` : retirer les spots expirés
- Step modifié : action_id == 8 → check gather_collective
- Si succès collectif, distribuer bonus aux participants

### 2.3 Action space étendu

n_actions passe de 8 (4 move + 4 vocalize) à **9** (+ 1 gather_collective).

LineageAgent gère automatiquement l'expansion via `cfg.cooperative.enabled`.

### 2.4 Visualisation

Les gather_spots apparaissent dans `obs` comme un nouveau canal :
- `gather_view[i,j] = 1.0` si tile (i,j) a un spot actif

→ `obs_dim = 6 * (2r+1)² + 3 + vocab_dim` (6 canaux au lieu de 5)

---

## 3. Régime `coordination_collective`

Nouvelle entrée dans `overnight_v8b1.py` :

```python
elif regime == "coordination_collective":
    # V8-C3 : V8-C2.b'' + gather_collective
    rows, cols = 40, 40
    n_agents = 20
    max_pop = 80
    food_respawn_lambda = 0.55
    metabolism = 0.3
    biome_cfg = ... (idem coordination_hard)
    cooperative_cfg = CooperativeConfig(
        enabled=True,
        min_partners_adjacent=1, signal_window_ticks=5,
        bonus_energy=30.0, spawn_density=0.02,
    )
```

---

## 4. Protocole expérimental

1. **Smoke 5k** : vérifier que les agents trouvent et exploitent les
   gather_spots (pas extinction, naissance d'au moins 1 gather success)
2. **Témoin 30k** : régime coordination_collective standard
3. **Ablation 30k** : idem + `--vocalize-disable-after 15000`
4. **Comparaison** : `compare_ablation.py`
5. **Si H1** : multi-seed 3×30k pour solidifier
6. **Si H0** : revoir design ou conclure que l'architecture ne supporte
   pas la coordination linguistique

---

## 5. Pièges anticipés

1. **Cold start aggravé** : action_space=9 et obs_dim+121. Le DQN doit
   apprendre plus. Augmenter epsilon_decay_steps si nécessaire.
2. **Gather_spots trop denses** : si 5 % des tiles ont un spot, le bonus
   devient trivial. Calibrer à 1-2 %.
3. **Gather_spots trop rares** : si <0.5 %, les agents ne s'y intéressent
   jamais. Calibrer empiriquement.
4. **bonus_energy trop élevé** : si +30 > food_value (18), les agents
   se concentrent uniquement sur gather, ignorent food, écosystème
   déséquilibré. À calibrer.
5. **min_partners=1** = 2 agents au total. Tester aussi min_partners=2
   (3 agents) pour augmenter la pression coordination.

---

## 6. Critères de succès V8-C3

V8-C3 livré (`v0.8.17-alpha`) si :

- [ ] Smoke 5k : pas d'extinction, ≥1 gather success
- [ ] Témoin 30k : pop stable, langage actif
- [ ] Ablation 30k : verdict mesuré (H0/H1/H2)
- [ ] Documentation finding ou null

Le résultat est **scientifiquement positif dans les 3 cas** :
- H1 → preuve d'émergence linguistique fonctionnelle
- H0 → contrainte architecturale forte révélée
- H2 → effet partiel à explorer

---

## 7. Non-objectifs V8-C3

- Pas de signal "venez ici" hardcodé
- Pas de reward direct sur l'usage du langage pour gather
- Pas de spécialisation forcée (V8-C4 plus tard)
- Pas de plus de 2 agents requis (commencer simple)

---

## 8. Curriculum V8-C3 (ajout 2026-05-25 après smoke initial)

**Leçon smoke 5k (régime hard) : extinction t=1310.** Un agent non
entraîné ne peut pas survivre dans un monde durci ET découvrir une
nouvelle action coopérative simultanément. On scinde C3 en 3 phases.

### C3a — apprendre gather dans un monde viable

Hypothèse : pour qu'un comportement coopératif émerge via Q-learning,
il faut que le signal de reward (succès gather) atteigne le replay
buffer assez souvent. Si succès trop rares → bruit pur, pas
d'attracteur comportemental stable.

**Paramètres curriculum C3a' (amplifié après smoke 10k → 29 succès) :**

| Paramètre | Valeur | Justification |
|---|---|---|
| `hidden_food` | False | Pas de cumul de difficultés |
| `max_pop` | 100 | Pop stable, évite pression écologique excessive |
| `respawn_threshold` | 2 | Resurrection des lignées éteintes |
| `winter_factor` | 0.5 | Saison dure mais survivable |
| `start_energy` | 220 | Réservoir initial confortable |
| `bonus_energy` | **80** | Signal de reward gros |
| `spawn_lambda` | **1.0** | Densité haute (1 spot/tick attendu) |
| `decay_ticks` | **100** | Fenêtre large (les agents ont le temps) |
| `max_active_spots` | **80** | Pas de saturation |

### C3b — durcir progressivement (après C3a validé)

À implémenter SI critère d'entrée OK :
- `hidden_food=True`
- `winter_factor=0.45`
- `bonus_energy=50` (retour valeur originelle)
- `spawn_lambda=0.5`

### C3c — ablation langage (après C3b validé)

`disable_vocalize_after_tick=15000` sur run 30k pour test interventionnel.

---

## 9. Critère d'entrée curriculum (ne PAS ablater trop tôt)

Avant ablation, **TOUS** ces seuils doivent être atteints :

1. `gather_successes_total ≥ 50` (signal pas du pur bruit)
2. ≥ 3 lignées vivantes (diversité préservée)
3. Pas d'extinction (écosystème stable)

Si non → on raffine le curriculum, on ne mesure pas une mécanique
non maîtrisée.

---

## 10. 4 métriques d'émergence coopérative (V8-C3 télémétrie)

Module : `aetherlife/world/cooperative_metrics.py`
Tracker : `CooperativeMetricsTracker` (observationnel pur, 0 influence
sur le training).

### Métrique 1 — clustering_pre_success

Pour chaque succès gather, compter agents vivants dans rayon manhattan
≤ 3 du spot. Tendance Q4-Q1 sur le run : si positive, **convergence
spatiale apprise**.

### Métrique 2 — vocalize_to_gather_delay

Pour chaque succès, mesurer le délai entre la dernière vocalize d'un
participant et le succès. Tendance Q4-Q1 négative = **apprentissage
réel d'un protocole**.

### Métrique 3 — token_entropy_pre_success

Distribution des tokens vocalisés dans une fenêtre de 5 ticks AVANT
chaque succès. Si un token devient dominant (share > 0.5, entropy
basse), c'est un candidat **proto-signal coopératif**.

### Métrique 4 — success_chains

Compter les cascades : succès consécutifs dans une fenêtre de 10 ticks.
`max_chain_len`, `n_cascade_successes ≥ 3` indiquent que **un succès
déclenche d'autres comportements coopératifs derrière** (preuve d'un
attracteur, pas juste de la chance).

### Critères de "proto-coordination" pour C3b → C3c

Avant ablation langage, on veut voir AU MOINS 2 sur 4 :
- [1] `trend(Q4-Q1) > +0.5` sur clustering
- [2] `trend(Q4-Q1) < -0.5` sur delay
- [3] `dominant_share > 0.40` sur tokens pre-success
- [4] `n_cascade_successes / n_successes > 0.20`

Si 0/4 → C3a' encore à raffiner. Si 4/4 → ablation très informative.
