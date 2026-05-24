# V8-B1.6 — Spéciation par contraintes incompatibles

> **Statut** : spec post-finding V8-B1.5.
> **Date** : 2026-05-24
> **Prédécesseur** : V8-B1.5 (`v0.8.5-alpha`, commit `3139783`).
> **Finding scientifique central** :
>
> > **« V8-B1.5 a prouvé que la cognition de lignée est trop forte pour de
> > simples biomes décoratifs. La diversité nécessite des niches avec
> > contraintes incompatibles, pas seulement des variations
> > d'environnement. »**

---

## 0. Diagnostic du run 100k V8-B1.5 niches

Verdict : **échec sur 4/4 critères biodiversité**.

| Critère | Cible | Résultat 100k niches |
|---|---|---|
| ≥ 3 lignées vivantes finales | ≥ 3 | **1** ✗ |
| Aucune lignée > 70 % | < 70 % | **100 %** (root=4) ✗ |
| KL inter-lignées > 0.05 | > 0.05 | **0.0** ✗ |
| Lifespan stable Q4 > Q1 | Q4 > Q1 | artefact statistique (vivants Q4 incomplets) |

Chronologie :
- t=5000 : 16 lignées
- t=10000 : 2 lignées
- t=**20000 : 1 lignée** ← monoculture
- t=20k→100k : stable 80 vivants, 1 lignée, 100 % dominance

**Insight inattendu** : la lignée gagnante a colonisé **les 4 biomes simultanément**. Lifespan énorme (Q3 mean = 17 297 ticks). Le cerveau partagé a généralisé à TOUTES les niches au lieu de se spécialiser.

→ Les biomes décoratifs (Voronoi avec différentiels metabolism/food_lambda 0.15-2.0) ne sont **pas des contraintes**. Un bon généraliste les contourne.

---

## 1. Hypothèse V8-B1.6

Pour qu'une lignée se spécialise, il faut **lui rendre l'autre stratégie coûteuse au point de ne pas être viable**. Sans malus hors niche, le monopole reste optimal.

Mécanisme proposé : **biome_affinity hérité**.

```
Chaque lignée fondatrice → tire un biome_affinity ∈ {PLAIN, FOREST, DESERT, TUNDRA}
À la reproduction → enfant hérite l'affinity du parent (1:1, pas de mutation)

Quand un agent est SUR son biome_affinity :
  - metabolism × 0.7 (-30 %)
  - food_value × 1.3 (+30 %)
  - reproduction permise
  - movement_cost normal

Quand un agent est HORS son biome_affinity :
  - metabolism × 1.5 (+50 %)
  - food_value × 0.7 (-30 %)
  - reproduction IMPOSSIBLE
  - movement_cost × 2.5
```

Conséquence théorique : un agent FOREST-affinity ne peut **pas** monopoliser le DESERT, car son métabolisme y est trop élevé et il ne peut pas s'y reproduire. La meilleure stratégie devient « rester dans son biome » → 4 lignées spécialisées coexistent.

---

## 2. Architecture

### 2.1 `_AgentState.biome_affinity: int | None`

```python
@dataclass
class _AgentState:
    ...
    # V8-B1.6 — affinity héritée (None si désactivé)
    biome_affinity: int | None = None
```

### 2.2 Tirage initial à reset()

```python
if self.cfg.biomes.enabled and self.cfg.biomes.affinity_enabled:
    for a in self._agents:
        a.biome_affinity = int(self._spawn_rng.integers(0, 4))
```

Distribution uniforme initial : N/4 agents par affinity (en moyenne).

### 2.3 Héritage 1:1 à la reproduction

Dans `_try_reproductions()` :
```python
child.biome_affinity = parent.biome_affinity
```

Pas de mutation. L'affinity est une **espèce écologique** stable.

### 2.4 Bonus/malus dans step()

```python
tile_biome = int(self._biome_map[new_r, new_c])
biome_p = biome_params_for(tile_biome, self.cfg.biomes)
local_metabolism = self.cfg.metabolism * biome_p.metabolism_factor
local_food_value = self.cfg.food_value * biome_p.food_value_factor

# V8-B1.6 — bonus/malus affinity
if agent.biome_affinity is not None:
    if tile_biome == agent.biome_affinity:
        local_metabolism *= self.cfg.biomes.in_affinity_metabolism  # 0.7
        local_food_value *= self.cfg.biomes.in_affinity_food_value   # 1.3
    else:
        local_metabolism *= self.cfg.biomes.out_affinity_metabolism  # 1.5
        local_food_value *= self.cfg.biomes.out_affinity_food_value  # 0.7
```

### 2.5 Movement cost amplifié hors affinity

Le `movement_cost` du biome est appliqué comme **metabolism additionnel** au moment du move. Hors affinity, multiplié par `out_affinity_movement_mult` (× 2.5).

### 2.6 Reproduction biome-locked

Dans `_try_reproductions()`, ajouter le check :
```python
if parent.biome_affinity is not None:
    parent_tile = int(self._biome_map[parent.pos])
    if parent_tile != parent.biome_affinity:
        continue  # repro impossible hors affinity
```

C'est le **gating dur**. Une lignée FOREST ne peut se reproduire qu'en forêt. Si toute la forêt meurt, la lignée s'éteint et son cerveau est perdu (sauf via seed bank = grace_ticks).

### 2.7 Worldgen équilibré

Modifier `generate_biome_map` pour **garantir** les 4 biomes représentés. Méthode : round-robin sur les seeds au lieu de random.

```python
# Au lieu de seed_biomes = rng.integers(0, 4, size=n)
# → forcer répartition équilibrée
n_per_type = max(1, n_seed_points // 4)
seed_biomes = np.array(
    [b for b in range(4) for _ in range(n_per_type)]
    + list(rng.integers(0, 4, size=max(0, n_seed_points - 4 * n_per_type))),
    dtype=np.int8,
)
rng.shuffle(seed_biomes)
```

---

## 3. Configuration

```python
@dataclass(frozen=True)
class BiomeConfig:
    enabled: bool = False
    n_seed_points: int = 8

    # V8-B1.6 — Affinities héritées
    affinity_enabled: bool = False
    in_affinity_metabolism: float = 0.7
    in_affinity_food_value: float = 1.3
    out_affinity_metabolism: float = 1.5
    out_affinity_food_value: float = 0.7
    out_affinity_movement_mult: float = 2.5
    reproduction_locked_to_affinity: bool = True

    # Biomes...
    # Worldgen...
```

---

## 4. Critères de succès V8-B1.6

V8-B1.6 livré (`v0.8.6-alpha`) si run 100k mode "speciation" produit :

- [ ] ≥ 3 lignées vivantes après 100k ticks
- [ ] Dominance max < 70 %
- [ ] ≥ 2 biome_affinities représentées en fin
- [ ] KL inter-lignées > 0.05
- [ ] Aucun biome (sur les 4) vide durablement (> 20 % du temps)
- [ ] Tests pytest ≥ 340 passed
- [ ] Tag git `v0.8.6-alpha`

---

## 5. Non-objectifs V8-B1.6

- **Pas** d'isolement spatial fort (continents séparés) — peut venir en V8-B1.7
- **Pas** de ressources multiples (fruits/cactus/fish) — V8-B2
- **Pas** d'assortative mating au niveau couple — V8-B2
- **Pas** de mutation d'affinity (l'affinity est strictement héritée)
- **Pas** de langage — B2 ou plus tard
- **Pas** de chunks infinis — B-monde

---

## 6. Plan TDD bite-sized

1. Étendre `BiomeConfig` avec champs affinity_*
2. Tests `BiomeConfig` validation des nouveaux params
3. `_AgentState.biome_affinity` champ + tests
4. `reset()` tirage initial uniforme + test distribution
5. `_try_reproductions()` héritage affinity + test
6. `_try_reproductions()` gating biome-locked + test
7. `step()` bonus/malus metabolism/food_value + test
8. `step()` movement_cost amplifié + test
9. Worldgen équilibré (round-robin) + test diversité
10. Mode CLI `speciation` dans launch_gui_v3 et overnight_v8b1
11. Run smoke 5k ticks puis 100k → verdict
