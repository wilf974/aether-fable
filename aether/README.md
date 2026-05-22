# Aether invariants — AetherLife

Catalogue d'invariants formels pour AetherLife, écrits en Aether v1.4 (test runner property-based via le MCP `mcp__aether__verify`).

## Catalogue V1

| ID | Fichier | Invariant |
|----|---------|-----------|
| I1 | `i1_energy_no_food.aether` | `0 ≤ result ≤ energy` après step sans manger |
| I2 | `i2_energy_with_food.aether` | `0 ≤ result ≤ max_energy` après step avec eat |
| I3 | `i3_terminated.aether` | `terminated ⟺ energy ≤ 0` |
| I4 | `i4_step_reward.aether` | `reward = -metabolism + (food_value si ate)` |
| I5 | `i5_clamp_pos.aether` | `0 ≤ result < dim` après step (clamping bord) |

## Vérifier

### Smoke harness (présence + format)

```bash
bash aether/verify_all.sh
```

### Validation property-based (via MCP Aether v1.4)

Chaque fichier est passé à `mcp__aether__verify` côté Claude Code. La validation s'exécute sur les `@example` et vérifie les `@invariant` sur chacun.

## Mirror Python

Chaque invariant a un mirror runtime dans `aetherlife/guardrails/invariants.py` testé par pytest. Les deux pistes (formelle Aether + runtime Python) doivent rester synchronisées.

## Convention de versioning

L'ajout d'un invariant `I_n+1` requiert :
1. Création du fichier `aether/invariants/iN_xxx.aether`.
2. Ajout du mirror Python.
3. Test pytest associé.
4. Mise à jour de ce README + `verify_all.sh`.
