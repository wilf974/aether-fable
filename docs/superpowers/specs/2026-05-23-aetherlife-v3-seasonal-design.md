# AetherLife V3 — Saisons + Météo + Food Regen Modulé — Design

**Date** : 2026-05-23
**Statut** : adopté
**Goal** : étendre V2 avec saisons cycliques + température 2D + food regen modulé, pour préparer le test de H3 (LSTM > MLP face à des régimes non-stationnaires).

## Périmètre V3

**Dans le scope** :
- `SeasonalConfig` : période saisonnière (default 200 ticks), 4 saisons, 4 facteurs lambda (spring=1.5, summer=1.0, autumn=1.2, winter=0.3).
- `SeasonClock` (via méthode `phase` / `season` sur l'env) : phase ∈ [0, 1).
- `WeatherField` : température 2D = gradient nord-sud + modulation saisonnière sinusoïdale, clampée dans `[temp_min, temp_max]`.
- `SeasonalMultiAgentFoodGrid` : étend `MultiAgentFoodGrid` avec :
  - Food regen modulé par saison (I11).
  - Metabolism × `cold_metabolism_factor` dans cellules < `cold_threshold`°C.
- Observation per-agent étendue : `[self, others, food, energy, phase, local_temp_normalisée]` (3 H W + 3 dims).
- 3 nouveaux invariants Aether (I9 phase ∈ [0,1), I10 temp clampée, I11 lambda saisonnier ≥ 0) + mirror Python.
- GUI V3 dédiée avec **heatmap température** + label saison courant.

**Hors scope V3 (V3.5+)** :
- DRQN (LSTM) wrapper from MW_IA et benchmark MLP vs LSTM (V3.5).
- Dangers actifs (storms, droughts) — reportés V3.5/V4.
- Ressources multiples (herbes vs fruits) — V3.5.
- Migration apprise (faisable en V3 si on entraîne un LSTM).

## Architecture

```
aetherlife/world/seasonal_grid.py    # SeasonalConfig + SeasonalMultiAgentConfig + SeasonalMultiAgentFoodGrid
aetherlife/viz/pygame_viewer_v3.py   # GUI dédiée + heatmap température
scripts/launch_gui_v3.py             # CLI
aether/invariants/i9_*.aether        # 3 nouveaux invariants
```

## Configuration V3 (defaults)

| Paramètre | Default | Sens |
|---|---|---|
| `season_period` | 200 | 50 ticks par saison |
| `spring_lambda_factor` | 1.5 | abondance printanière |
| `summer_lambda_factor` | 1.0 | nominale |
| `autumn_lambda_factor` | 1.2 | fruits secondaires |
| `winter_lambda_factor` | 0.3 | rareté hivernale |
| `temp_min` / `temp_max` | -10 / 30 | range Celsius |
| `temp_gradient_amplitude` | 10 | spread nord-sud |
| `seasonal_amplitude` | 15 | spread saisonnier |
| `cold_threshold` | 5.0 | cellules ≤ 5°C |
| `cold_metabolism_factor` | 1.5 | metabolism × 1.5 dans le froid |

## Observation per-agent V3

```
obs[i] = concat(
    self_pos_one_hot,        # rows * cols
    others_alive_one_hot,    # rows * cols
    food_one_hot,            # rows * cols
    [my_energy_normalized, season_phase, local_temp_normalisée],  # 3
)
```

Dim = `3 * rows * cols + 3` = 3075 pour 32×32.

## Invariants Aether V3 (déjà validés)

| ID | Fichier | Propriété |
|----|---------|-----------|
| I9 | `i9_season_phase.aether` | `0 ≤ phase < 1` |
| I10 | `i10_clamp_temp.aether` | `temp_min ≤ temp ≤ temp_max` |
| I11 | `i11_seasonal_lambda.aether` | `seasonal_lambda(base, factor) ≥ 0` |

## Critères de succès V3

1. `pytest -q` → tous tests V1+V1.5+V2+V3 verts (cible 150+).
2. `bash aether/verify_all.sh` → 11 OK.
3. GUI V3 lance et affiche correctement saison + heatmap température.
4. Phénomène saisonnier observable : alive_rate plus haute en spring/autumn qu'en winter.

## Suite V3.5+ (sortie de scope V3)

- **V3.5** : DRQN wrapper (mw_ia.agents.recurrent_dqn) + benchmark MLP vs LSTM sur env V3 → test H3.
- **V4** : reproduction + lineage + PBT (population-based training).
- **V5** : émergence coopération (MAPPO, QMIX) sur env V3 enrichi.
