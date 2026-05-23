# AetherLife

Plateforme RL pour l'étude expérimentale des comportements émergents dans des écosystèmes artificiels procéduraux.

**Statut** : V2 (`v0.2.0-alpha`) — multi-agent IDQN + tragedy of the commons démontrée.

## Vision

Construire une plateforme scientifique où des agents RL autonomes apprennent à survivre, explorer, coopérer, migrer et s'adapter dans un environnement procédural dynamique. Les comportements doivent **émerger** via l'apprentissage, pas être codés à la main.

Hypothèses testées au fil des versions :
- **H1** — émergence de la coopération sous scarcité.
- **H2** — spécialisation et niches sans architecture multi-tête.
- **H3** — adaptation aux régimes (saisons) avec mémoire LSTM.
- **H4** — co-évolution prédateur/proie (Lotka-Volterra).
- **H5** — transmission culturelle entre agents naïfs et experts.

Voir le design complet dans `docs/superpowers/specs/`.

## V1 — Solo Forager

Un agent unique sur grille 16×16 statique, gère son énergie via food respawning, meurt si énergie ≤ 0.

| Composant | Fichier |
|---|---|
| Config frozen dataclass | [`aetherlife/config.py`](aetherlife/config.py) |
| Env core (numpy pur) | [`aetherlife/world/food_grid.py`](aetherlife/world/food_grid.py) |
| Wrapper Gymnasium | [`aetherlife/env/single_agent_env.py`](aetherlife/env/single_agent_env.py) |
| Invariants Python mirror | [`aetherlife/guardrails/invariants.py`](aetherlife/guardrails/invariants.py) |
| Agent random | [`aetherlife/agents/random_agent.py`](aetherlife/agents/random_agent.py) |
| Agent greedy oracle | [`aetherlife/agents/greedy_agent.py`](aetherlife/agents/greedy_agent.py) |
| Agent DQN (wrap MW_IA) | [`aetherlife/agents/dqn_agent.py`](aetherlife/agents/dqn_agent.py) |
| Best-checkpoint tracker | [`aetherlife/training/best_checkpoint.py`](aetherlife/training/best_checkpoint.py) |
| DQN runner + assessment | [`aetherlife/training/dqn_runner.py`](aetherlife/training/dqn_runner.py) |
| GUI live pygame | [`aetherlife/viz/pygame_viewer.py`](aetherlife/viz/pygame_viewer.py) |
| Invariants Aether (I1-I5) | [`aether/invariants/`](aether/invariants/) |

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows : Git Bash
pip install -e ".[dev]"

# V1.5+ : DQN MW_IA + PyTorch CUDA
pip install -e "../MW_IA"
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install pygame-ce
```

## Smoke baseline

```bash
python scripts/run_baseline.py --episodes 100
```

Résultat attendu sur defaults :

```
AetherLife V1 baseline — episodes=100  grid=16x16  max_steps=1000
  RandomAgent   : survival= 87.0%  lifespan=  879.3  reward= +3983.6  food=243.5
  GreedyAgent   : survival=100.0%  lifespan= 1000.0  reward= +8509.6  food=475.5
```

## V1.5 — Entraîner un DQN

Defaults rapides (smoke ~50 s GPU, best ~20 %) :

```bash
python scripts/train_dqn.py --episodes 300 --device cuda
```

**Recette gagnante** (best assessment 90 %, ~8 min RTX 3060) :

```bash
python scripts/train_dqn.py \
    --episodes 1500 --device cuda \
    --hidden 256 256 --epsilon-decay-steps 40000 --target-sync-steps 300 \
    --lr 5e-4 --batch-size 256 \
    --assess-every 25 --assess-episodes 10 --patience 25
```

L'entraînement utilise `mw_ia.agents.dqn.DQNAgent` (réutilise QNetwork + ReplayBuffer +
DQNTrainer), avec **assessment greedy périodique** (toutes les 25 ép), **best-checkpoint
tracking** et **early stopping** (patience N évaluations sans amélioration).

Le checkpoint best est sauvé dans `checkpoints/dqn_best.pt`.

## V2 — Multi-agent IDQN

Entraîner un IDQN shared-weights sur 16 agents :

```bash
python scripts/train_v2_idqn.py --n-agents 16 --episodes 300 --device cuda
```

### Tragedy of the commons — density sweep

Démontrer l'effondrement collectif sous densité haute (random policy, instantané) :

```bash
python scripts/v2_density_sweep.py
```

Résultats observés sur defaults (grille 32×32, density 0.05, respawn λ=1.0, 10 ép/N) :

```
   N |  alive_rate |  mean_life |  mean_food | food/agent
   2 |      45.0% |      292.4 |       51.3 |     25.65
   4 |      35.0% |      235.7 |       81.3 |     20.32
   8 |      46.2% |      296.8 |      180.6 |     22.57
  16 |      25.0% |      209.8 |      214.4 |     13.40
  32 |      14.7% |      154.5 |      250.0 |      7.81
```

→ ressource individuelle (food/agent) chute de 25.7 (N=2) à 7.8 (N=32) : **70 % de perte**, alive_rate divisé par 3.

## GUI live

```bash
python scripts/launch_gui.py
python scripts/launch_gui.py --dqn-checkpoint checkpoints/dqn_best.pt   # avec DQN
```

**Contrôles** :
- `SPACE` pause / reprise
- `R` reset épisode
- `A` switch agent (Greedy ↔ Random ↔ DQN si checkpoint fourni)
- `↑` / `↓` accélérer / ralentir
- `Q` ou `ESC` quitter

## Vérifier les invariants

### Smoke harness (présence + format)

```bash
bash aether/verify_all.sh
```

### Property-based (via MCP Aether v1.4 dans Claude Code)

Chaque fichier `aether/invariants/iN_*.aether` est validé par `mcp__aether__verify`. La V1 a 23 examples + 13 assertions, toutes vertes.

### Mirror Python (pytest)

```bash
pytest tests/guardrails/ -v
```

## Tests

```bash
pytest -q
```

Attendu V2 : **122 tests verts** (V1 : 72, V1.5 : 18, V2 : 11 ma_grid + 8 idqn + 3 ma_runner + 10 I6-I8 mirror = 32).

## Catalogue d'invariants V1

| ID | Fichier Aether | Mirror Python | Propriété |
|----|---|---|---|
| I1 | `i1_energy_no_food.aether` | `guardrails.invariants.energy_no_food` | `0 ≤ result ≤ energy` |
| I2 | `i2_energy_with_food.aether` | `guardrails.invariants.energy_with_food` | `0 ≤ result ≤ max_energy` |
| I3 | `i3_terminated.aether` | `guardrails.invariants.is_terminated` | `result ⟺ energy ≤ 0` |
| I4 | `i4_step_reward.aether` | `guardrails.invariants.step_reward` | `reward = -metabolism + food_value · ate` |
| I5 | `i5_clamp_pos.aether` | `guardrails.invariants.clamp_pos` | `0 ≤ result < dim` |
| I6 | `i6_pop_after_deaths.aether` | `guardrails.invariants.pop_after_deaths` | `0 ≤ result ≤ n_alive_before` |
| I7 | `i7_energy_gained.aether` | `guardrails.invariants.energy_gained` | `result = n_food_eaten × food_value` |
| I8 | `i8_total_ids_emitted.aether` | `guardrails.invariants.total_ids_emitted` | `result = n_alive + n_dead` (pas de réutilisation d'id) |

## Roadmap

| V | Périmètre | Statut |
|---|---|---|
| **V1** | Solo Forager + baselines + invariants I1-I5 | ✅ livré (`v0.1.0-alpha`) |
| **V1.5** | Wrap DQN MW_IA + GUI live pygame + best-checkpoint + assessment | ✅ livré (`v0.1.5-alpha`) |
| **V2** | Multi-agent IDQN shared-weights + invariants I6-I8 + tragedy of the commons | ✅ livré (`v0.2.0-alpha`) |
| V2 | Multi-agent independent (IDQN), tragedy of the commons | À venir |
| V3 | Saisons + météo + food regen dynamique (LSTM vs MLP) | À venir |
| V4 | Reproduction + lineage + PBT | À venir |
| V5 | Émergence coopération (MAPPO/QMIX) | À venir |
| V6 | Prédation + Lotka-Volterra | À venir |
| V7 | Co-évolution + transmission culturelle | À venir |
| V8 | Plateforme publique + dataset + preprint | À venir |

Voir le design détaillé dans `docs/superpowers/specs/2026-05-23-aetherlife-v1-solo-forager-design.md`.

## Stack technique

- Python 3.13+
- NumPy 1.26+, Gymnasium 0.29+
- pytest pour les tests, hypothesis pour le property-based
- Aether v1.4 (MCP) pour la vérification formelle des invariants
- Réutilise l'infra RL de [MW_IA](../MW_IA/) (DQN, ConvDQN, DRQN, Double DQN — intégration V1.5+).

## Licence

Pas encore décidée (probablement MIT ou Apache-2.0 à la V8 publique).
