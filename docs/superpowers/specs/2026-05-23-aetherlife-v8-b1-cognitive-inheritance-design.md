# V8-B1 — Évolution cognitive minimale (cerveau par lignée + héritage de poids)

> **Statut** : spec en cours de validation, pré-implémentation.
> **Date** : 2026-05-23
> **Prédécesseur** : V7 — Traits héritables (livré `v0.7.0-alpha`, 283 tests).
> **Position dans la roadmap globale** (AetherLife = plateforme évolutive H24) : phase B1 du plan d'architecture illimitée (cf. messages user 2026-05-23).

---

## 0. TL;DR

Remplacer `SmartHeuristicAgent` par un agent RL héritable. Pour chaque lignée
(racine = `root_ancestor_id`), un **cerveau partagé** (réseau DQN compact)
décide les actions de tous les agents vivants de cette lignée. À la
reproduction, l'enfant hérite des poids du parent avec mutation gaussienne.

Pas de chunks infinis. Pas de langage. Pas de LLM. Pas de communication.
Just survie + héritage cérébral + sélection naturelle.

**Critère de succès** : sur un run de 10000 ticks (mode darwinien),
au moins une lignée doit atteindre `n_alive ≥ 20`, `gen ≥ 5`, et
afficher un comportement comportementalement distinct de la baseline
SmartHeuristic (mesuré par MSE des trajectoires).

---

## 1. Pourquoi maintenant

État au handoff V7 :
- env saisonnier + 6 mécaniques (food, build, family, cache, plant, traits)
- 283 tests verts, 21 invariants Aether
- `SmartHeuristicAgent` = oracle déterministe : produit des comportements visibles
  mais ne **prouve pas l'intelligence**.
- Traits héritables fonctionnent (sélection mesurée +0.14 build_bias en régime
  darwinien sur 10000 ticks, dominance 90% de root=3).

Avant d'ouvrir chunks infinis (B-monde) ou langage (B-comm), il faut prouver
que l'évolution cognitive **fonctionne mécaniquement** :
- héritage de poids stable (pas de divergence numérique)
- sélection lignée-niveau visible (lignée pauvre s'éteint, lignée riche
  propage)
- comportements adaptatifs émergents (pas juste random)
- stabilité long terme (pas de collapse)

**Si B1 échoue, tout le reste s'effondre. Si B1 réussit, l'ambition long
terme (civilisation, langage, espace) devient ouverte.**

---

## 2. Architecture détaillée

### 2.1 `LineageBrain` — un cerveau par racine ancestrale

```python
@dataclass
class LineageBrain:
    """Réseau RL partagé par tous les agents d'une lignée."""
    root_ancestor_id: int
    network: nn.Module               # MLP compact (hidden 64×64 par défaut)
    target_network: nn.Module        # target net DQN
    replay_buffer: ReplayBuffer       # buffer partagé par la lignée
    optimizer: torch.optim.Optimizer
    epsilon: float                    # exploration courante
    global_step: int                  # nombre de transitions vues
    n_alive_agents: int               # combien d'agents utilisent ce cerveau
    parent_brain_id: int | None       # pour audit lignée
```

### 2.2 `LineageRegistry` — map root → brain

```python
class LineageRegistry:
    """Indexe les cerveaux par root_ancestor_id."""
    def __init__(self, cfg: BrainConfig, device: str = "cuda"):
        self._brains: dict[int, LineageBrain] = {}
        self.cfg = cfg
        self.device = device

    def get_or_create(self, root_id: int, parent_brain: LineageBrain | None = None) -> LineageBrain:
        if root_id in self._brains:
            return self._brains[root_id]
        # Création : copy parent + mutation, OU init random si pas de parent
        ...

    def cull_dead_lineages(self, alive_roots: set[int]) -> None:
        """Free les cerveaux dont la lignée s'est éteinte."""
        ...
```

### 2.3 Héritage de poids à la reproduction

```python
def inherit_with_mutation(
    parent_network: nn.Module,
    mutation_std: float,
    rng: np.random.Generator,
    device: str = "cuda",
) -> nn.Module:
    """Clone le réseau parent + ajoute bruit gaussien sur tous les poids."""
    child = copy.deepcopy(parent_network)
    with torch.no_grad():
        for p in child.parameters():
            noise = torch.tensor(
                rng.normal(0.0, mutation_std, size=p.shape),
                device=device, dtype=p.dtype,
            )
            p.add_(noise)
    return child
```

**Quand muter ?**
- À la reproduction `parent → child` SI le child démarre une **nouvelle lignée**
  (pas le même `root_ancestor_id` que parent). Sinon, child rejoint le même
  cerveau partagé.
- V5.2 hérite déjà `root_ancestor_id = parent.root_ancestor_id`. Donc tous
  les descendants partagent le cerveau du fondateur.
- **Mutation à appliquer** : quand le **fondateur** d'une lignée se reproduit
  pour la première fois, son cerveau est **forké** en deux : un pour lui-même
  (inchangé), un pour la descendance (légère mutation). Cela permet
  divergence inter-lignée tout en gardant cohésion intra-lignée.

**Décision V8-B1 (simplification)** : on évite le fork. Une lignée = un seul
cerveau. La mutation se fait au **niveau lignée** : pour chaque nouvelle
lignée racine créée (au reset, ou au fork futur), bruitage léger des poids
du parent_brain.

### 2.4 Observation locale

Vu l'objectif "cerveau minimum viable", on **réduit** l'observation par rapport
au full env :

| Composant | Dim | Calcul |
|---|---|---|
| Vision food | 11×11 = 121 | one-hot food dans fenêtre |
| Vision nests | 11×11 = 121 | one-hot nests dans fenêtre |
| Vision plants | 11×11 = 121 | one-hot plants matures dans fenêtre |
| Vision agents | 11×11 = 121 | one-hot autres agents |
| Self state | 4 | énergie, faim, fatigue, age_norm |
| Saison | 4 | one-hot (spring/summer/autumn/winter) |
| **Total** | **492** | |

Pas de coord absolue, pas de temp_field : observation **égocentrique**, comme
un vrai animal. C'est ce qui permettra la généralisation aux chunks infinis
plus tard (l'agent ne sait pas où il est dans le monde, juste ce qu'il voit).

### 2.5 Actions

V8-B1 : on garde les actions atomiques V7 (NORTH/EAST/SOUTH/WEST) **plus**
les actions implicites V5+ (build/plant/cache deposit/withdraw qui se
déclenchent par règle quand l'agent est dans la bonne config).

**Pas encore** d'actions explicites supplémentaires (mate / vocalize / etc.) —
ça arrive en B2/B3.

Action space : `Discrete(4)`. Comme V2.

### 2.6 Reward shaping

Inchangé V7 :
- `r = -metabolism + food_value × ate`
- `+ rest_bonus si sur nid`
- `+ cache_withdrawal_amount si retrait`
- `- death_penalty si mort`

C'est volontairement **dense + non dirigé**. Pas de "+10 si tu plantes" — la
plantation n'est récompensée que par sa conséquence (récolte ultérieure).

### 2.7 Cycle d'entraînement

```
   Pour chaque tick :
     1. Pour chaque agent vivant :
        a. lookup son brain via root_ancestor_id
        b. brain.act(obs) → action
     2. env.step(actions)
     3. Pour chaque agent vivant après step :
        a. brain.observe(prev_obs, action, reward, next_obs, done)
        b. brain.train() si replay buffer plein
     4. Reproduction si éligible :
        a. enfant hérite root_ancestor_id du parent
        b. enfant utilise le brain du parent (pas de mutation, même lignée)
     5. Cull lineages éteintes
        a. brain release GPU memory
```

### 2.8 Stratégie d'init

- Au reset, chaque agent fondateur (0..N-1) crée sa propre `LineageBrain`
  avec init **random orthogonal** (PyTorch default).
- Si mutation_std=0 et 0 fondateurs => le système est équivalent à V7
  (testable).

---

## 3. Configuration

```python
@dataclass(frozen=True)
class BrainConfig:
    """V8-B1 — Configuration du cerveau par lignée."""

    enabled: bool = False              # compat V7 et avant

    # Architecture
    hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "relu"

    # RL hyperparams
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_capacity: int = 50_000
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    target_sync_steps: int = 300

    # Héritage
    mutation_std: float = 0.02         # std du bruit sur les poids
    inherit_at_lineage_creation: bool = True

    # Observation
    vision_radius: int = 5             # 11×11
    include_season: bool = True

    # Practical
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self) -> None:
        ...validation
```

---

## 4. Invariants Aether à ajouter

| ID | Propriété | Statut |
|---|---|---|
| I22 | `weight_after_mutation` reste fini si poids parent fini et std<10 | À écrire |
| I23 | `child_lineage_inherits_root` : root_ancestor_id transitif | À vérifier (déjà OK V5.2 en principe) |
| I24 | `brain_lookup_consistent` : 2 agents même root → même brain | À écrire |
| I25 | `cull_safe` : free un brain ne casse pas la lignée vivante | À écrire |

---

## 5. Tests V8-B1 prévus

### 5.1 Tests unitaires (`tests/agents/test_lineage_brain.py`)
- `test_brain_creation_random_init`
- `test_brain_inherit_with_mutation_changes_weights`
- `test_brain_inherit_with_zero_mutation_is_identity`
- `test_brain_act_returns_valid_action`
- `test_brain_observe_pushes_to_buffer`
- `test_brain_train_decreases_loss`

### 5.2 Tests registry (`tests/agents/test_lineage_registry.py`)
- `test_registry_get_or_create_new_lineage`
- `test_registry_reuses_existing_brain`
- `test_registry_cull_dead_lineage_frees_memory`
- `test_registry_handles_concurrent_lineages`

### 5.3 Tests intégration (`tests/integration/test_v8b1_smoke.py`)
- `test_v8b1_smoke_no_crash` : 500 ticks sans erreur
- `test_v8b1_lineage_brain_persists_across_generations`
- `test_v8b1_lineages_diverge_with_mutation`

### 5.4 Benchmark
- `scripts/bench_v8b1.py` : run 10k ticks, mesure :
  - `survival_rate_by_lineage`
  - `mean_lifespan_by_lineage`
  - `food_efficiency = total_food_eaten / total_lifespan`
  - `behavioral_distinctness = MSE(trajectory_lineage_A, trajectory_lineage_B)`
  - `cognitive_evolution = best_brain_eval_at_t=10000 vs t=0`

---

## 6. Pièges anticipés

1. **GPU memory explose** si trop de lignées en parallèle.
   Mitigation : cap `max_concurrent_lineages = 20`. Cull aggressif.
2. **Replay buffer froid** pour lignées qui débutent.
   Mitigation : `start_step=1000` avant first train.
3. **Mutation détruit le savoir** si std trop élevée.
   Mitigation : `mutation_std=0.02` initial (très conservateur).
4. **Multi-thread CUDA** entre brains.
   Mitigation : V8-B1 = single-thread, optim seulement quand train batch.
5. **Comportement initial pire que random** car NN non entraîné.
   Mitigation : `epsilon_start=0.5` pour exploration forte au début.
6. **PyTorch sur CPU si CUDA indispo**.
   Mitigation : fallback CPU détecté automatiquement.

---

## 7. Plan TDD bite-sized

Voir `docs/superpowers/plans/2026-05-23-aetherlife-v8-b1-plan.md` (à créer).

Granularité : 8-12 commits, chacun rouge → vert → refactor.

1. `BrainConfig` dataclass + validation
2. `LineageBrain` minimal (init, act, observe, save/load)
3. Inheritance avec mutation (PyTorch deepcopy + jitter)
4. `LineageRegistry` (get_or_create, cull)
5. Wiring dans `SeasonalMultiAgentFoodGrid.step()`
6. Observation locale (égocentrique)
7. Invariants Aether I22-I25 + Python mirror
8. Tests intégration smoke
9. Benchmark script
10. GUI panneau "Cerveaux actifs" dans pygame_viewer

---

## 8. Critères de validation V8-B1

V8-B1 est livré (`v0.8.0-alpha`) quand :

- [ ] Tests : `pytest -q` → ≥ 300 passed (283 V7 + ~20 nouveaux)
- [ ] Aether : 25 invariants validés via `mcp__aether__verify`
- [ ] Smoke 1k ticks mode darwinien : pas de crash, lignées qui évoluent
- [ ] Benchmark 10k ticks : ≥ 1 lignée atteint gen ≥ 5 ET n_alive ≥ 20
- [ ] Behavioral distinctness mesurable entre 2 lignées (MSE > 0.1)
- [ ] GUI affiche n cerveaux actifs + lignée focus
- [ ] Tag git `v0.8.0-alpha` + commit avec rapport

---

## 9. Non-objectifs V8-B1 (explicitement)

- **Pas** de chunks infinis (B-monde)
- **Pas** d'action `vocalize` / langage (B-comm)
- **Pas** de tech tree émergent (C)
- **Pas** de persistance Postgres (D)
- **Pas** de frontend web (F)
- **Pas** de NN > 1M params (compactness first)
- **Pas** de policy gradients (PPO/A2C) — DQN suffit pour le moment
- **Pas** de communication inter-lignée
- **Pas** de fork de cerveau intra-lignée (1 brain par root strict)

Ces objectifs reviendront en B2, B3, C, D, E, F respectivement.

---

## 10. Risques + plan B

| Risque | Plan B |
|---|---|
| RL trop instable, lignées dégénèrent en 100 ticks | Fallback : init poids depuis SmartHeuristic via behavioral cloning |
| GPU memory OOM avec >20 lignées | Sharding sur CPU + batch inference |
| Aucune divergence visible entre lignées | Augmenter mutation_std × 5 + relancer |
| Comportements pire que random | Réviser obs_dim (peut-être trop riche), simplifier |
| Tests trop lents (>30s) | Réduire `n_agents=4` + `max_steps=200` pour smoke |
