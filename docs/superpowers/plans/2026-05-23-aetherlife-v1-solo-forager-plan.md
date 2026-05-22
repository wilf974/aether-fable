# AetherLife V1 — Solo Forager — Plan TDD

**Goal** : implémenter V1 minimal (env + 2 baselines + invariants) en TDD, tests pytest verts, smoke script tournant.

**Architecture** : `aetherlife/` package pur Python+numpy + wrapper Gymnasium ; `aether/` invariants formels (déjà livrés) ; `guardrails/` mirror Python.

**Tech Stack** : Python 3.13, NumPy, Gymnasium 0.29+, pytest.

---

## Task 1 — `config.py` : FoodGridConfig

**Files** :
- Create: `aetherlife/config.py`
- Test: `tests/world/test_config.py`

### Steps

- [ ] **1.1** Test rouge `tests/world/test_config.py::test_default_config` (instanciation defaults).
- [ ] **1.2** Test rouge `test_invalid_metabolism_raises` (metabolism ≤ 0 doit `ValueError`).
- [ ] **1.3** Test rouge `test_invalid_food_density_raises` (density hors [0, 1]).
- [ ] **1.4** Impl `FoodGridConfig` frozen dataclass + `__post_init__` validation.
- [ ] **1.5** `pytest tests/world/test_config.py -v` → 3 verts.
- [ ] **1.6** Commit `feat(config): FoodGridConfig frozen dataclass`.

## Task 2 — `guardrails/invariants.py` : mirror Python des 5 Aether

**Files** :
- Create: `aetherlife/guardrails/invariants.py`, `aetherlife/guardrails/exceptions.py`
- Test: `tests/guardrails/test_invariants.py`

### Steps

- [ ] **2.1** Test rouge : 5 tests `test_energy_no_food`, `test_energy_with_food`, `test_is_terminated`, `test_step_reward`, `test_clamp_pos` qui appellent les fonctions avec les mêmes `@example` que les fichiers Aether.
- [ ] **2.2** Impl 5 fonctions strictement miroir des Aether (mêmes signatures, mêmes formules).
- [ ] **2.3** Impl `InvariantViolationError` (extends `RuntimeError`).
- [ ] **2.4** `pytest tests/guardrails/ -v` → 5 verts.
- [ ] **2.5** Commit `feat(guardrails): 5 invariants Python mirror Aether I1-I5`.

## Task 3 — `world/food_grid.py` : core env

**Files** :
- Create: `aetherlife/world/food_grid.py`
- Test: `tests/world/test_food_grid.py`

### Steps

- [ ] **3.1** Test rouge `test_reset_returns_valid_obs` (reset, vérifie shape obs et valeurs initiales).
- [ ] **3.2** Test rouge `test_step_decreases_energy_by_metabolism` (step sans food → energy -= metabolism).
- [ ] **3.3** Test rouge `test_step_eats_food_when_on_food_cell` (step vers food → energy += food_value, food disparaît).
- [ ] **3.4** Test rouge `test_terminated_when_energy_zero` (forcer energy=0 → terminated=True).
- [ ] **3.5** Test rouge `test_truncated_at_max_steps`.
- [ ] **3.6** Test rouge `test_action_clamp_at_borders` (move au bord ne sort pas de la grille).
- [ ] **3.7** Test rouge `test_food_respawn_after_eat` (eat puis quelques ticks → la food count revient ≥ initial - 1).
- [ ] **3.8** Test rouge `test_seed_reproducibility` (même seed → mêmes positions food).
- [ ] **3.9** Impl `FoodGrid` class avec `reset(seed)`, `step(action) → (obs, reward, terminated, truncated, info)`.
- [ ] **3.10** `pytest tests/world/test_food_grid.py -v` → 8 verts.
- [ ] **3.11** Commit `feat(world): FoodGrid env core (V1)`.

## Task 4 — `env/single_agent_env.py` : wrapper Gymnasium

**Files** :
- Create: `aetherlife/env/single_agent_env.py`
- Test: `tests/env/test_single_agent_env.py`

### Steps

- [ ] **4.1** Test rouge `test_gym_env_compliance` (utilise `gym.utils.env_checker.check_env`).
- [ ] **4.2** Test rouge `test_observation_space_bounds` (Box[0, 1]^513).
- [ ] **4.3** Test rouge `test_action_space_discrete_4`.
- [ ] **4.4** Impl `SoloForagerEnv(gym.Env)` qui wrap `FoodGrid`.
- [ ] **4.5** `pytest tests/env/ -v` → 3 verts.
- [ ] **4.6** Commit `feat(env): SoloForagerEnv Gymnasium-compatible`.

## Task 5 — `agents/base.py + random_agent.py + greedy_agent.py`

**Files** :
- Create: `aetherlife/agents/base.py`, `aetherlife/agents/random_agent.py`, `aetherlife/agents/greedy_agent.py`
- Test: `tests/agents/test_agents.py`

### Steps

- [ ] **5.1** Test rouge `test_random_agent_returns_valid_action`.
- [ ] **5.2** Test rouge `test_greedy_agent_moves_toward_nearest_food`.
- [ ] **5.3** Test rouge `test_greedy_agent_handles_no_food` (renvoie une action valide même si pas de food).
- [ ] **5.4** Impl `Agent` Protocol + `RandomAgent` + `GreedyAgent`.
- [ ] **5.5** `pytest tests/agents/ -v` → 3 verts.
- [ ] **5.6** Commit `feat(agents): RandomAgent + GreedyAgent baselines`.

## Task 6 — Smoke script + README + commit final

**Files** :
- Create: `scripts/run_baseline.py`, `README.md`, `CLAUDE.md`

### Steps

- [ ] **6.1** Impl `scripts/run_baseline.py` : run N épisodes pour `RandomAgent` et `GreedyAgent`, print stats (survival rate, mean lifespan, mean total reward).
- [ ] **6.2** Run `python scripts/run_baseline.py --episodes 100` → vérifier `GreedyAgent > RandomAgent` sur survival rate.
- [ ] **6.3** Run `bash aether/verify_all.sh` → 5 OK.
- [ ] **6.4** Run `pytest -q` → tous verts.
- [ ] **6.5** Écrire README.md (intro, install, run baseline, structure).
- [ ] **6.6** Écrire CLAUDE.md (état V1, conventions, procédures usuelles).
- [ ] **6.7** Commit `feat(v1): smoke baseline + README + CLAUDE.md` + tag `v0.1.0-alpha`.

---

## Définition of Done V1

- ✅ `pytest -q` → tous tests verts.
- ✅ `bash aether/verify_all.sh` → 5 OK.
- ✅ `python scripts/run_baseline.py --episodes 100` :
  - `GreedyAgent` survival rate ≥ 95 %.
  - `RandomAgent` < `GreedyAgent`.
- ✅ Aucun warning Python.
- ✅ Tag `v0.1.0-alpha` posé.
