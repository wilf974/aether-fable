# AetherLife V3.7 — MA Spatial Benchmark — Design

**Date** : 2026-05-23
**Statut** : adopté

## Objectif

> **Prouver que CNN+DDQN dépasse MLP quand la perception spatiale devient nécessaire.**

V3.6 a montré que sur env simple (16×16, density 5 %, single-agent), MLP suffit et CNN coûte 3× plus en compute pour le même résultat. V3.7 crée le **terrain où l'archi spatiale est justifiée** : multi-agent + 32×32 + density basse + saisons contraignantes.

## Périmètre V3.7

**Dans le scope** :
- `IndependentConvDQNAgent` : IDQN shared-weights utilisant `ConvDQNAgent` MW_IA V2-W (Double DQN par défaut) au lieu du MLP V2.
- `multi_agent_conv_runner.py` : MA training loop avec observations 2D (dict[int, (4, R, C)]).
- `observation_2d_dict()` sur `SeasonalMultiAgentFoodGrid` (filtré agents vivants).
- Script benchmark `scripts/v3_7_benchmark_ma_seasonal_complex.py` : MLP IDQN vs CNN+DDQN IDQN multi-seed.
- **Évaluation held-out stricte** : assessment sur seeds `base=100_000+i` jamais utilisés en training.

**Hors scope V3.7 (V3.8/V4)** :
- IDQN MLP sur env SAISONNIER : V2 IndependentDQNAgent utilise FoodGrid (non-saisonnier). Pour comparer rigoureusement, le baseline MLP V3.7 utilise `MultiAgentFoodGrid` (V2) — env strictement plus simple que le CNN+DDQN qui voit les saisons. **C'est un biais en faveur de MLP** : si CNN gagne malgré cela, c'est encore plus probant.
- Multi-seed n≥3 statistique : V3.7 livre l'infra, smoke fait sur 1 seed pour valider la pipeline.
- FOV per-agent (V2.5+).

## Configuration V3.7 (defaults benchmark)

| Paramètre | Default | Sens |
|---|---|---|
| `rows`, `cols` | 32, 32 | grille plus large que V3.6 (16×16) |
| `n_agents` | 16 | tragedy of commons potentielle |
| `initial_food_density` | 0.02 | très basse (vs 0.05 V3.6) → pression compétitive |
| `food_respawn_lambda` | 1.0 | regen modeste |
| `season_period` | 80 | 20 ticks par saison |
| `winter_lambda_factor` | 0.2 | rare en hiver (vs 0.3 default V3) |
| `spring_lambda_factor` | 2.0 | boom printanier (vs 1.5 default) |
| `summer_lambda_factor` | 1.0 | nominale |
| `autumn_lambda_factor` | 1.2 | fruits |
| `max_steps` | 300 | episode plus long pour traverser plusieurs saisons |
| `episodes` | 400 | budget compute raisonnable |

## Architecture CNN+DDQN V3.7

Réutilise `mw_ia.agents.conv_dqn.ConvDQNAgent` (V2-W de MW_IA) :
- ConvQNetwork : Conv(4→32, k=3, p=1) → ReLU → Conv(32→64) → ReLU → Flatten → Linear(256) → Linear(4)
- Double DQN : `next_action = argmax(online(s'))`, `q_next = target(s')[next_action]`
- AMP désactivé (use_amp=False) pour stabilité
- Shared weights : 1 réseau, N inférences

Observation 2D 4 canaux (par agent) :
- canal 0 : self_position one-hot
- canal 1 : others_alive positions
- canal 2 : food_mask
- canal 3 : temperature normalisée

## Critères de succès V3.7

1. `pytest -q` tous verts (~195).
2. Smoke benchmark tourne sans crash sur 200 ép.
3. **CNN+DDQN ≥ MLP IDQN sur alive_rate held-out** (≥3 pp de différence ou identique).
4. Si CNN+DDQN < MLP : analyse + tuning (lr, conv_channels, batch_size) puis itérer.

## Suite V3.8+ (sortie de scope)

- **V3.8** : multi-seed n=3 sur recette V3.7 pour statistiques solides.
- **V4** : reproduction + lineage + PBT (population-based training).
- **V5** : émergence coopération via MAPPO / QMIX.
