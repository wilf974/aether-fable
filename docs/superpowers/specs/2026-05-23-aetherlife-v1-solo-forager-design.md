# AetherLife V1 — Solo Forager — Design

**Date** : 2026-05-23
**Statut** : adopté
**Goal** : valider l'infra AetherLife (env + agents baseline + invariants Aether + tests + GUI optionnelle) sur un environnement minimal single-agent (1 agent, énergie, food, métabolisme, mort).

## Périmètre

**Dans le scope V1** :
- Grille 2D statique avec cellules `libre` et `food`.
- 1 agent unique avec énergie bornée `[0, max_energy]`.
- 4 actions cardinales : `NORTH`, `SOUTH`, `EAST`, `WEST`.
- Auto-eat : si la cellule cible contient de la food, food consommée, `+food_value` énergie.
- Food respawn dynamique (Poisson par tick) pour maintenir densité approximative.
- Reward : `-metabolism` par step + `food_value` si eat + `-death_penalty` à la mort.
- Terminaison : `energy ≤ 0` (mort). Truncation : `step ≥ max_steps`.
- 2 agents baseline : `RandomAgent` (sanity), `GreedyAgent` (oracle direction vers food la plus proche).
- 5 invariants Aether (I1-I5) + mirror Python runtime.
- Tests pytest sur env, invariants, agents.
- Smoke script `scripts/run_baseline.py`.

**Hors scope V1 (reportés V1.5+)** :
- DQN / RL training (intégration MW_IA en V1.5).
- Obstacles dans la grille (reporté V2).
- Multi-agent (V2).
- Saisons / météo (V3).
- GUI live (V1.5).

## Architecture

```
aetherlife/
├── config.py              # FoodGridConfig frozen dataclass + __post_init__ validation
├── world/
│   └── food_grid.py       # FoodGrid (core env, pure Python+numpy, sans dépendance Gymnasium)
├── env/
│   └── single_agent_env.py # SoloForagerEnv (Gymnasium-compatible wrapper)
├── guardrails/
│   ├── invariants.py      # 5 fonctions mirror Python de I1-I5
│   └── exceptions.py      # InvariantViolationError
└── agents/
    ├── base.py            # Protocol Agent
    ├── random_agent.py
    └── greedy_agent.py
```

## Configuration V1 (defaults)

| Paramètre | Default | Justification |
|---|---|---|
| `rows`, `cols` | 16, 16 | grille assez petite pour rapidité, assez grande pour la stratégie |
| `max_energy` | 100 | borne haute |
| `start_energy` | 50 | démarre à mi-énergie pour ne pas favoriser greedy initial |
| `metabolism` | 1 | -1 par step |
| `food_value` | 20 | un eat = ~20 steps de survie |
| `death_penalty` | 50 | pénalité significative mais pas écrasante |
| `initial_food_density` | 0.05 | ~12 cellules food sur 16×16 = 256 |
| `food_respawn_lambda` | 0.5 | en moyenne 0.5 food spawn par tick |
| `max_steps` | 1000 | truncation après 1000 ticks |

## Observation (V1)

`observation = concat(position_one_hot, food_grid_flatten, energy_normalized)` :
- `position_one_hot` : `rows * cols` (256)
- `food_grid_flatten` : `rows * cols` (256)
- `energy_normalized` : 1 (energy / max_energy)
- Total : 513 dim.

Aligné sur `encode_procedural_observation` de MW_IA V2-X.

## Action space

`Discrete(4)` : `0=NORTH, 1=SOUTH, 2=WEST, 3=EAST` (cohérent MW_IA `Action` enum).

## Invariants Aether V1 (déjà validés)

| ID | Fichier | Mirror Python |
|----|---------|---------------|
| I1 | `i1_energy_no_food.aether` | `guardrails.invariants.energy_no_food` |
| I2 | `i2_energy_with_food.aether` | `guardrails.invariants.energy_with_food` |
| I3 | `i3_terminated.aether` | `guardrails.invariants.is_terminated` |
| I4 | `i4_step_reward.aether` | `guardrails.invariants.step_reward` |
| I5 | `i5_clamp_pos.aether` | `guardrails.invariants.clamp_pos` |

Tous les 5 ont été validés via `mcp__aether__verify` (23 examples + 13 assertions invariants).

## Critères de succès V1

1. `pytest -q` → **tous tests verts** (cible ≥ 20 tests V1).
2. `bash aether/verify_all.sh` → **5 OK**.
3. `python scripts/run_baseline.py` → `GreedyAgent` survit ≥ 95 % de 100 épisodes de 1000 steps.
4. `RandomAgent` < `GreedyAgent` (sanity, sinon env est cassé ou trop facile).
5. Aucun warning Python sur un run complet.

## Décisions techniques

- **`FoodGrid` indépendant de Gymnasium** : core env reste pur Python+numpy, le wrapper Gymnasium est en couche au-dessus. Permet de tester l'env sans dépendance Gym.
- **`SoloForagerEnv(gym.Env)`** : implémentation Gymnasium standard (reset/step retournent tuples 4-arity et 5-arity respectivement).
- **`np.random.Generator`** : RNG distribués par sous-système — un `env_rng` pour la grille initiale, un `spawn_rng` pour le respawn. Évite la dérive de seed à l'ajout de modules.
- **Pas d'obstacles V1** : V1 minimal = food + agent. Les obstacles ajoutent du chemin-finding non essentiel pour valider l'infra survie.
- **Invariants Python = miroir 1:1 des Aether** : exactement les mêmes signatures et formules. Toute divergence est un bug.

## Risques V1 (et mitigation)

| Risque | Mitigation |
|---|---|
| Greedy ne converge pas (env trop hostile) | tunable `food_value`/`metabolism` au runtime, defaults choisis pour balance |
| Tests flaky (randomness) | seed fixé en tests (`env_seed=0`) |
| Drift Aether ↔ Python | tests pytest qui appellent les fonctions guardrails et comparent aux examples Aether |

## Suite V1.5 (sortie de scope V1)

- Wrapper MW_IA DQN/ConvDQN (réutilisation directe `mw_ia.agents.dqn.DQNAgent`).
- GUI live (port de `mw_ia.gui`).
- Best-checkpoint tracking (port V2-V — préalable à V2).
