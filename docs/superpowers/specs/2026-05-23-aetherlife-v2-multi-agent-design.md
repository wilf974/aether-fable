# AetherLife V2 — Multi-Agent Independent — Design

**Date** : 2026-05-23
**Statut** : adopté
**Goal** : étendre V1.5 à N agents indépendants sur la même grille pour observer la **tragedy of the commons** (effondrement collectif sous densité haute).

## Périmètre V2

**Dans le scope** :
- Grille 32×32 (vs 16×16 V1) — plus grande pour accueillir multiple agents.
- N agents simultanés (sweep N ∈ {4, 8, 16, 32}).
- Chacun agent a : `agent_id` (stable), `position`, `energy`, `alive` flag.
- Une seule cellule peut contenir plusieurs agents (V2 minimal, simplifie).
- Auto-eat : si plusieurs agents sur même cellule food au tick d'eat, **ordre par `agent_id` ascendant** (déterministe).
- Reward purement **individuel** (pas de partage — V5 introduit le partage).
- Action interface PettingZoo-style : `step(actions: dict) → 5 dicts`.
- Observation per-agent : `[self_pos_one_hot, others_pos_one_hot, food_one_hot, my_energy_normalized]`.
- 3 nouveaux invariants Aether (I6-I8) + mirror Python.
- IDQN avec **shared weights** par défaut (un seul réseau, N instances d'inférence) — économise VRAM, biais d'apprentissage homogène acceptable V2.
- Smoke : entraîner shared-IDQN, mesurer mean lifespan inter-agents.

**Hors scope V2 (V2.5/V3)** :
- FOV per-agent (partial observability) — V2 utilise observation globale.
- Conflits sur food (déterministe par id-order suffit V2).
- Reproduction (V4).
- Coopération / partage explicite (V5).
- Visualisation MA dans la GUI (V2.5).

## Architecture

```
aetherlife/
├── world/
│   ├── food_grid.py            # V1 inchangé
│   └── multi_agent_grid.py     # [V2] MultiAgentFoodGrid
├── env/
│   ├── single_agent_env.py     # V1 inchangé
│   └── multi_agent_env.py      # [V2] MultiAgentForagerEnv (PettingZoo-style)
├── agents/
│   ├── ...                     # V1/V1.5 inchangés
│   └── independent_dqn.py      # [V2] IndependentDQNAgent (shared weights par défaut)
├── guardrails/
│   └── invariants.py           # + pop_after_deaths, energy_gained, total_ids_emitted [V2]
├── training/
│   ├── ...                     # V1.5 inchangés
│   └── multi_agent_runner.py   # [V2] MA training loop
```

## Configuration V2 (defaults)

```python
@dataclass(frozen=True)
class MultiAgentForagerConfig:
    rows: int = 32
    cols: int = 32
    n_agents: int = 16
    max_energy: float = 100.0
    start_energy: float = 50.0
    metabolism: float = 1.0
    food_value: float = 20.0
    death_penalty: float = 50.0
    initial_food_density: float = 0.05
    food_respawn_lambda: float = 1.0   # plus haut V2 (plus d'agents = plus de food)
    max_steps: int = 1000
    spawn_radius: int = 0              # 0 = positions aléatoires partout
```

## Observation per-agent

```
obs[i] = concat(
    self_pos_one_hot_i,      # (rows * cols,)  : 1 sur la cellule de i
    others_alive_one_hot,    # (rows * cols,)  : somme des positions des j ≠ i vivants
    food_one_hot,            # (rows * cols,)  : 1 par cellule food
    [my_energy / max_energy] # (1,)
)
```

Dim = `3 * rows * cols + 1` = `3 * 1024 + 1 = 3073` pour 32×32.

## Action space

`Discrete(4)` par agent, identique V1 (`NORTH, SOUTH, WEST, EAST`).

## Step semantics

1. Pour chaque agent vivant **dans l'ordre `agent_id` ascendant** :
   1. Compute `new_pos` via `clamp_pos`.
   2. Si food at `new_pos` : agent consomme (`+food_value`, clampé à `max_energy`), food retirée.
   3. Sinon : `energy -= metabolism`.
   4. Si `energy ≤ 0` : `alive = False`, `terminated[id] = True`, reward += `-death_penalty`.
2. `step_count += 1`.
3. Food respawn (Poisson `food_respawn_lambda`) sur cellules libres.
4. Build obs_dict pour agents vivants au début du step (terminés inclus, vu qu'ils ont reçu leur reward final).
5. Truncated[id] = True si `step_count ≥ max_steps` et pas terminated.

## Invariants Aether V2 (déjà validés)

| ID | Fichier | Propriété |
|----|---------|-----------|
| I6 | `i6_pop_after_deaths.aether` | `0 ≤ n_alive_after ≤ n_alive_before` |
| I7 | `i7_energy_gained.aether` | `total_energy_gained = n_food_eaten × food_value` |
| I8 | `i8_total_ids_emitted.aether` | `n_alive + n_dead = total_ids_emitted` (pas de réutilisation) |

## IndependentDQNAgent (shared weights)

- 1 seul réseau partagé entre les N agents.
- `act(obs_dict) → action_dict` : forward sur stack des observations vivants.
- `observe(transitions_dict)` : pousse chaque transition individuellement dans le replay buffer commun.
- Hyperparams : mêmes defaults que V1.5 recette gagnante.

## Critères de succès V2

1. `pytest -q` → tous tests V1+V1.5+V2 verts.
2. `bash aether/verify_all.sh` → 8 OK.
3. `python scripts/train_v2_idqn.py --n-agents 16 --episodes 500 --device cuda` :
   - Smoke entraînement complet sans crash.
   - Mean lifespan agent ≥ 200 steps (sur grille 32×32, density 0.05, respawn λ=1.0).
4. Sweep densité N ∈ {4, 8, 16, 32} montre une courbe de survie inversée (haute N → effondrement).

## Risques V2 + mitigation

| Risque | Mitigation |
|---|---|
| VRAM explose à N=32 + obs 3073-dim | Shared weights (1 réseau), batch agrégé par tick |
| Convergence MA non-stationnaire (politiques co-évoluent) | Shared weights atténue (homogénéité) ; pour V5 → MAPPO |
| Variance énorme inter-agents | Plotter individual lifespan distribution + agg metrics |
| Tests flaky (randomness MA) | Seed fixé + tests sur petits N (2-3 agents) |
| Episode boundary ambigu MA | Définition : episode finit quand `step_count >= max_steps` OU tous morts |

## Suite V2.5+ (sortie de scope V2)

- **V2.5** : FOV per-agent (partial observability) + visualisation MA dans la GUI.
- **V3** : saisons + météo + food regen dynamique (test LSTM > MLP).
- **V4** : reproduction + lineage + PBT.
- **V5** : émergence coopération (MAPPO, QMIX) sur env V2 enrichi.
