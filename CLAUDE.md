# CLAUDE.md — AetherLife

> **À lire en premier** par Claude dans une nouvelle session du projet AetherLife.

## Règles de comportement

- Répondre en **français**.
- Invoquer les skills **superpowers** (brainstorming, writing-plans, TDD, executing-plans) quand pertinent.
- Utiliser le MCP **aether** activement pour vérifier les invariants RL avant de les implémenter en Python. Pattern V1 : 5 invariants Aether validés puis mirror Python.
- Utiliser le MCP **context7** pour la doc des librairies (Gymnasium, PyTorch, etc.).
- TaskCreate/TodoWrite pour piloter les tâches multi-étapes.

## État au handoff (2026-05-23 — V1 livré `v0.1.0-alpha`)

V1 Solo Forager complet :
- **72 tests pytest verts** (17 config + 27 guardrails + 14 food_grid + 7 env + 7 agents).
- **5 invariants Aether v1.4 validés** (23 examples + 13 assertions invariants).
- **5 invariants Python mirror** runtime-checkés.
- **Smoke baseline** : `GreedyAgent` 100 % survival × 100 ép, `RandomAgent` 87 % (Greedy bat clairement Random).
- **Spec + plan TDD** dans `docs/superpowers/specs/` et `docs/superpowers/plans/`.

### Composants V1 livrés

| Module | Fichier | Rôle |
|---|---|---|
| Config | `aetherlife/config.py` | `FoodGridConfig` frozen dataclass + validation |
| Env core | `aetherlife/world/food_grid.py` | Grille 2D, food, énergie, respawn, observation 513-dim |
| Wrapper Gym | `aetherlife/env/single_agent_env.py` | `SoloForagerEnv(gym.Env)` (conforme Gymnasium 0.29+) |
| Guardrails | `aetherlife/guardrails/invariants.py` | Mirror Python de I1-I5 (toute divergence vs Aether = bug) |
| Exceptions | `aetherlife/guardrails/exceptions.py` | `InvariantViolationError` |
| Random | `aetherlife/agents/random_agent.py` | Baseline sanity uniforme |
| Greedy | `aetherlife/agents/greedy_agent.py` | Oracle Manhattan vers food la plus proche |
| Smoke | `scripts/run_baseline.py` | Compare Random vs Greedy sur N épisodes |
| Aether | `aether/invariants/i1-i5.aether` | Catalogue formel (validés via `mcp__aether__verify`) |
| Harness | `aether/verify_all.sh` | Smoke présence + format |

### Catalogue d'invariants V1

| ID | Fichier Aether | Fonction Python | Propriété formelle |
|----|---|---|---|
| I1 | `i1_energy_no_food.aether` | `energy_no_food` | `0 ≤ result ≤ energy` |
| I2 | `i2_energy_with_food.aether` | `energy_with_food` | `0 ≤ result ≤ max_energy` |
| I3 | `i3_terminated.aether` | `is_terminated` | `result ⟺ energy ≤ 0` |
| I4 | `i4_step_reward.aether` | `step_reward` | `reward = -metabolism + food_value · ate` |
| I5 | `i5_clamp_pos.aether` | `clamp_pos` | `0 ≤ result < dim` |

## Environnement machine (vérifié 2026-05-23)

- OS : Windows 11 Pro 10.0.26200
- Shell Claude Code : Git Bash (Bash tool), PowerShell via `powershell.exe -NonInteractive -Command`
- Python : 3.13.12 (`py -3.13` ou venv local `.venv/`)
- Dépendances V1 : `numpy>=1.26`, `gymnasium>=0.29`, `pytest`, `hypothesis`
- Réutilisation : projet voisin `../MW_IA/` (RL infra V2-W livrée, integration V1.5+)

### Activer le venv

```bash
source .venv/Scripts/activate
```

## Procédures usuelles

### Lancer les tests
```bash
source .venv/Scripts/activate && pytest -q
```
Attendu : **72 passed**.

### Smoke baseline
```bash
python scripts/run_baseline.py --episodes 100
```
Attendu : GreedyAgent 100 % survival, RandomAgent ~87 %.

### Vérifier les invariants
```bash
bash aether/verify_all.sh
```
Attendu : 5 OK.

Pour la validation property-based formelle (Aether v1.4), passer chaque fichier `.aether` à `mcp__aether__verify` côté Claude Code.

## Conventions adoptées (héritage MW_IA + spécifiques V1)

- **Frozen dataclasses** : toutes les configs sont immutables et validées en `__post_init__`.
- **Pas d'héritage entre configs** : chaque config est explicit et complète (Python `dataclass(frozen=True)`).
- **`numpy.random.Generator` distribués** : `env_rng` et `spawn_rng` séparés par sous-système (cf. `FoodGrid.__init__`). Pas de `np.random.seed` global.
- **Mirror strict Aether ↔ Python** : toute fonction d'invariant Python a une contrepartie 1:1 dans `aether/invariants/`. Les `@example` Aether sont reproduits comme paramètres pytest.
- **TDD bite-sized** : test rouge → impl → test vert → commit. Pattern utilisé pour V1.

## Pièges connus V1

1. **`SoloForagerEnv.observation_space` clamping** : la dim 513 inclut un canal d'énergie normalisée `[0, 1]`. Si on augmente max_energy au runtime, l'observation reste dans `[0, 1]`. Mais si on retire le clamp dans `energy_with_food` (ex. via une variante reward shaping), elle peut sortir → tests env_checker Gymnasium tomberont.

2. **`food_respawn_lambda` interaction avec `initial_food_density`** : si initial_density=0 et respawn=0, l'env est trivialement perdant. Le smoke baseline utilise density=0.05 + lambda=0.5 (équilibre raisonnable).

3. **RandomAgent à 87 % survival** sur defaults : pas un bug, conséquence de food respawn généreux + grille petite. Pour un vrai écart RL→baselines en V1.5, baisser density à 0.02 et lambda à 0.1.

4. **Position de départ doit être unique** : `_initial_food_layout` exclut explicitement `start_position` du pool de food (test `test_start_position_never_has_food`).

5. **Gymnasium `check_env`** : OK avec `skip_render_check=True`. Le render `ansi` n'est pas standard render_mode Gymnasium (juste un debug helper).

## Roadmap (lien Spec)

Voir `docs/superpowers/specs/2026-05-23-aetherlife-v1-solo-forager-design.md` pour la roadmap V0 → V8 complète.

**Prochaine étape par défaut** : V1.5 = intégrer MW_IA DQN (et idéalement V2-V best-checkpoint préalable dans MW_IA) pour comparer un agent appris vs Greedy.

## Mémoires persistantes liées

- `~/.claude/projects/C--Users-Wilfred/memory/MEMORY.md` (index global)
- `~/.claude/projects/C--Users-Wilfred/memory/projet_mw_ia.md` (projet voisin, infrastructure RL)

## Instructions pour la prochaine session

1. Lire ce CLAUDE.md + `docs/superpowers/specs/2026-05-23-aetherlife-v1-solo-forager-design.md`.
2. Smoke test rapide :
   ```bash
   source .venv/Scripts/activate && pytest -q && bash aether/verify_all.sh
   ```
   Attendu : 72 passed + 5 OK.
3. Aligner avec l'utilisateur sur le périmètre V1.5 (DQN wrap + GUI) ou V2 (multi-agent).
4. Cycle complet pour tout nouveau sous-projet :
   - `superpowers:brainstorming` → cerner intent, scope, contraintes
   - Écrire la spec dans `docs/superpowers/specs/YYYY-MM-DD-<sub-projet>-design.md`
   - Concevoir + valider les nouveaux invariants Aether via `mcp__aether__verify`
   - `superpowers:writing-plans` → plan TDD bite-sized
   - `superpowers:subagent-driven-development` ou inline TDD selon scope.
