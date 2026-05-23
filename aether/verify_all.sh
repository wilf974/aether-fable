#!/usr/bin/env bash
# verify_all.sh — smoke harness pour le catalogue d'invariants AetherLife.
# La vraie validation property-based passe par le MCP mcp__aether__verify côté Claude Code.
# Ce script vérifie uniquement la présence et le format des fichiers .aether.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVARIANTS_DIR="${DIR}/invariants"

EXPECTED=(
    "i1_energy_no_food.aether"
    "i2_energy_with_food.aether"
    "i3_terminated.aether"
    "i4_step_reward.aether"
    "i5_clamp_pos.aether"
    "i6_pop_after_deaths.aether"
    "i7_energy_gained.aether"
    "i8_total_ids_emitted.aether"
    "i9_season_phase.aether"
    "i10_clamp_temp.aether"
    "i11_seasonal_lambda.aether"
)

missing=0
for f in "${EXPECTED[@]}"; do
    path="${INVARIANTS_DIR}/${f}"
    if [[ ! -f "${path}" ]]; then
        echo "MISSING: ${f}"
        missing=$((missing + 1))
    elif ! grep -q "(fn " "${path}"; then
        echo "MALFORMED (no fn): ${f}"
        missing=$((missing + 1))
    elif ! grep -q "@example" "${path}"; then
        echo "MALFORMED (no @example): ${f}"
        missing=$((missing + 1))
    elif ! grep -q "@invariant" "${path}"; then
        echo "MALFORMED (no @invariant): ${f}"
        missing=$((missing + 1))
    else
        echo "OK: ${f}"
    fi
done

if [[ ${missing} -gt 0 ]]; then
    echo ""
    echo "FAIL: ${missing} fichier(s) manquant(s) ou malformé(s)."
    exit 1
fi

echo ""
echo "PASS: ${#EXPECTED[@]} invariants présents et bien formés."
