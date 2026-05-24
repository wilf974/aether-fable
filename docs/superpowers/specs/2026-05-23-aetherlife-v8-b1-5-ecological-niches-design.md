# V8-B1.5 — Niches écologiques + anti-monoculture douce

> **Statut** : spec post-finding B1 (run 100k → monopole 100% root).
> **Date** : 2026-05-23
> **Prédécesseur** : V8-B1 (`v0.8.0-alpha`, commit `ed41b28` + viz `37591a3`).
> **Objectif scientifique** : démontrer la **coexistence cognitive stable** —
> ≥3 lignées vivantes après 100k ticks, aucune lignée >70% dominance, KL
> inter-lignées >0.05, lifespan moyen stable.

---

## 0. TL;DR

Le run V8-B1 100k ticks a produit un **monopole cognitif absolu** : 10 → 1
lignée en <15k ticks, dominance 100%. C'est un succès technique (RL
converge, héritage prouvé +77 % lifespan en test isolé) mais un échec
écologique : pas de coexistence.

Diagnostic : monde homogène + 1 ressource + 1 stratégie optimale = 1
winner. La solution n'est pas d'**empêcher** la sélection, mais de
**créer plusieurs façons de gagner**.

V8-B1.5 introduit :
1. **Biomes** (4 types) avec paramètres différents
2. **Worldgen Voronoi** procédural
3. **max_pop 30 → 100**, grid **64×64**
4. **Compétition locale** (metabolism modulé par densité voisine)
5. **Soft cull** : brains gardés N ticks après extinction de lignée

**Aucun langage. Aucun chunk infini.** Tout vient en B2/B-monde.

---

## 1. BiomeConfig

```python
class BiomeType(IntEnum):
    PLAIN = 0    # neutre, défaut
    FOREST = 1   # food abondante, mais coût mouvement
    DESERT = 2   # food rare, metabolism élevé
    TUNDRA = 3   # food très rare, metabolism très élevé, food_value haut


@dataclass(frozen=True)
class BiomeParams:
    metabolism_factor: float = 1.0   # multiplie metabolism du tile
    food_lambda_factor: float = 1.0  # multiplie food respawn
    food_value_factor: float = 1.0   # multiplie food_value mangé
    movement_cost: float = 1.0       # multiplie cost de move (Δenergy)
    passable: bool = True


@dataclass(frozen=True)
class BiomeConfig:
    enabled: bool = False
    plain: BiomeParams = BiomeParams()
    forest: BiomeParams = BiomeParams(
        food_lambda_factor=2.0, movement_cost=1.2,
    )
    desert: BiomeParams = BiomeParams(
        food_lambda_factor=0.3, metabolism_factor=1.3,
    )
    tundra: BiomeParams = BiomeParams(
        food_lambda_factor=0.15, metabolism_factor=1.5,
        food_value_factor=1.5,
    )
    n_seed_points: int = 8
```

### Stratégies optimales théoriques par biome

- **PLAIN** : équilibré, agents généralistes
- **FOREST** : food++ → spécialiste forager, peu de stockage
- **DESERT** : food-- mais survivre → planter intensivement (V6 cycle)
- **TUNDRA** : food très rare mais riche → migration entre tundra et autre, stockage massif

Hypothèse : 4 stratégies différentes = 4 lignées qui peuvent coexister
dans 4 niches. Si vrai, coexistence stable.

---

## 2. Worldgen Voronoi

```python
def generate_biome_map(rows, cols, cfg: BiomeConfig, seed: int) -> np.ndarray:
    """Generate biome_map via Voronoi avec n_seed_points seeds aléatoires."""
    rng = np.random.default_rng(seed)
    n = cfg.n_seed_points
    seed_coords = rng.uniform(0, max(rows, cols), size=(n, 2))
    seed_biomes = rng.integers(0, 4, size=n)  # type per seed
    biome_map = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            d = np.sum((seed_coords - [r, c]) ** 2, axis=1)
            biome_map[r, c] = seed_biomes[np.argmin(d)]
    return biome_map
```

Résultat : 4-8 zones contigües par biome, frontières naturelles.

---

## 3. Intégration dans SeasonalMultiAgentFoodGrid

### 3.1 État env
```python
class SeasonalMultiAgentFoodGrid:
    def __init__(self, cfg):
        ...
        self._biome_map: np.ndarray = ...  # 2D int8, rempli à reset()
```

### 3.2 Step modifié
```python
def step(self, actions):
    for agent in alive:
        ...
        # Lookup biome courant
        biome_id = self._biome_map[new_r, new_c]
        biome_params = self._biome_params_lookup(biome_id)

        # Metabolism modulé par biome
        local_metabolism = self.cfg.metabolism * biome_params.metabolism_factor
        # Si tile = tundra (cold), s'ajoute le cold_metabolism_factor de saisons
        if local_temp < self.cfg.seasonal.cold_threshold:
            local_metabolism *= self.cfg.seasonal.cold_metabolism_factor

        # Food value modulé par biome
        if ate:
            agent.energy = energy_with_food(
                agent.energy, local_metabolism,
                self.cfg.food_value * biome_params.food_value_factor,
                self.cfg.max_energy,
            )
```

### 3.3 Food spawn modulé
```python
def _spawn_food(self):
    for (r, c) in empty_cells:
        biome_id = self._biome_map[r, c]
        local_lambda = self.cfg.food_respawn_lambda * biome_params.food_lambda_factor
        # Poisson sampling avec λ ajusté
        ...
```

---

## 4. Observation égocentrique étendue

Avant : 4 canaux × 121 + 3 scalars = 487 dim.
Après V8-B1.5 : ajout canal biome (1 canal, valeur normalisée ∈ [0, 1]).

Nouveau dim : **5 × 121 + 3 = 608 dim**.

Encoding biome : `0.0/0.33/0.66/1.0` selon `BiomeType`.

Cette modification casse les cerveaux V8-B1 trained (dim mismatch). Donc
V8-B1.5 démarre avec cerveaux fresh — mais c'est OK car on veut tester
de coexistence dès le début.

---

## 5. Compétition locale

```python
@dataclass(frozen=True)
class CompetitionConfig:
    enabled: bool = False
    radius: int = 3              # voisins dans cercle de rayon R
    metabolism_per_neighbor: float = 0.03  # +3% metabolism par voisin proche
    max_factor: float = 2.0      # cap à 2× metabolism
```

Implémentation :
```python
n_neighbors = sum(
    1 for other in self._agents
    if other.alive and other.agent_id != agent.agent_id
    and manhattan(other.pos, agent.pos) <= ccfg.radius
)
crowd_factor = min(
    1.0 + n_neighbors * ccfg.metabolism_per_neighbor,
    ccfg.max_factor,
)
local_metabolism *= crowd_factor
```

Effet : agents dans des zones très peuplées brûlent plus d'énergie. Pousse
à l'**occupation de zones moins denses** = ouverture de niches spatiales.

---

## 6. Soft cull des brains

Avant V8-B1.5 : `cull_dead_lineages()` supprime immédiatement les brains
des lignées éteintes. Une lignée morte est perdue pour toujours.

Après V8-B1.5 :
```python
class LineageRegistry:
    def __init__(self, ..., grace_ticks: int = 5000):
        self._grace_ticks = grace_ticks
        self._extinction_ticks: dict[int, int] = {}

    def cull_dead_lineages(self, alive_roots, current_tick):
        dead = set(self._brains) - alive_roots
        # Marquer les nouvelles extinctions
        for r in dead:
            if r not in self._extinction_ticks:
                self._extinction_ticks[r] = current_tick
        # Free seulement après grace period
        to_free = [
            r for r, t in self._extinction_ticks.items()
            if (current_tick - t) >= self._grace_ticks
        ]
        for r in to_free:
            del self._brains[r]
            del self._extinction_ticks[r]
        return len(to_free)
```

Effet : si une lignée meurt à t=20000, son brain reste disponible
jusqu'à t=25000. Si entre temps un agent de cette lignée naît par
résurrection (rare mais possible si family inheritance + reset),
le savoir est récupéré.

---

## 7. Critères de succès V8-B1.5

V8-B1.5 livré (`v0.8.5-alpha`) quand :

- [ ] Tests : `pytest -q` ≥ 320 passed
- [ ] Aether : I22 toujours OK + nouveaux invariants biome
- [ ] **CRITÈRE BIODIVERSITÉ** : run 100k mode V8-B1.5 →
  - ≥ 3 lignées vivantes en fin
  - aucune lignée >70 % de la population
  - KL inter-lignées > 0.05
  - lifespan moyen Q4 > Q1 (héritage cognitif préservé)
- [ ] Viz GUI : biome_map visible (couleurs)
- [ ] Commit tag v0.8.5-alpha avec rapport

---

## 8. Non-objectifs V8-B1.5

- **Pas** de food types multiples (viande, fruit) → V8-B2 ou plus tard
- **Pas** de chunks infinis → V9 monde
- **Pas** de langage → après coexistence stable
- **Pas** d'élevage → V10
- **Pas** d'écriture → V12+

V8-B1.5 = "diversité écologique contrôlée" et rien d'autre.

---

## 9. Plan TDD bite-sized

1. `BiomeType`, `BiomeParams`, `BiomeConfig` + validation (RED→GREEN)
2. `generate_biome_map` worldgen Voronoi
3. Tests biome lookup + paramètres
4. `SeasonalMultiAgentConfig.biomes` field
5. `SeasonalMultiAgentFoodGrid` lookup biome dans step()
6. Food spawn modulé par biome
7. Observation égocentrique inclut canal biome
8. `CompetitionConfig` + competition radius
9. `LineageRegistry.cull_dead_lineages` soft (grace_ticks)
10. Viz GUI : biome_map en background colorize
11. Run 100k validation → verdict
