# AetherLife

Plateforme RL pour l'étude expérimentale des comportements émergents dans des écosystèmes artificiels procéduraux.

**Statut** : V1 (`v0.1.0-alpha`) — single-agent forager livré.

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
| Invariants Aether (I1-I5) | [`aether/invariants/`](aether/invariants/) |

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows : Git Bash
pip install -e ".[dev]"
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

Attendu V1 : **72 tests verts** (17 config + 27 guardrails + 14 food_grid + 7 env + 7 agents).

## Catalogue d'invariants V1

| ID | Fichier Aether | Mirror Python | Propriété |
|----|---|---|---|
| I1 | `i1_energy_no_food.aether` | `guardrails.invariants.energy_no_food` | `0 ≤ result ≤ energy` |
| I2 | `i2_energy_with_food.aether` | `guardrails.invariants.energy_with_food` | `0 ≤ result ≤ max_energy` |
| I3 | `i3_terminated.aether` | `guardrails.invariants.is_terminated` | `result ⟺ energy ≤ 0` |
| I4 | `i4_step_reward.aether` | `guardrails.invariants.step_reward` | `reward = -metabolism + food_value · ate` |
| I5 | `i5_clamp_pos.aether` | `guardrails.invariants.clamp_pos` | `0 ≤ result < dim` |

## Roadmap

| V | Périmètre | Statut |
|---|---|---|
| **V1** | Solo Forager + baselines + invariants I1-I5 | ✅ livré (`v0.1.0-alpha`) |
| V1.5 | Wrap DQN MW_IA + GUI live + best-checkpoint | À venir |
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
