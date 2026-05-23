"""Lance la GUI V3 saisonnière multi-agent AetherLife — Living Sandbox (V3.8).

Modes :
    easy   (default) — agents survivent longtemps, beaucoup de food
    normal           — defaults originaux V3
    hard             — env hostile (calibration V3.7 benchmark)

Usage :
    python scripts/launch_gui_v3.py
    python scripts/launch_gui_v3.py --mode normal
    python scripts/launch_gui_v3.py --mode hard --n-agents 32
    python scripts/launch_gui_v3.py --season-period 80
"""
from __future__ import annotations

import argparse

from aetherlife.viz.pygame_viewer_v3 import run_gui_v3
from aetherlife.world.construction import BuildConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import SeasonalConfig, SeasonalMultiAgentConfig


# Presets équilibrés pour observer les comportements sans tout tuer
MODE_PRESETS: dict[str, dict] = {
    "easy": dict(
        metabolism=0.5,
        food_value=12.0,
        start_energy=70.0,
        max_energy=120.0,
        death_penalty=10.0,
        initial_food_density=0.10,
        food_respawn_lambda=2.5,
        winter_factor=0.7,
        summer_factor=1.2,
        spring_factor=1.8,
        autumn_factor=1.3,
        max_steps=1500,
    ),
    "normal": dict(
        metabolism=1.0,
        food_value=20.0,
        start_energy=50.0,
        max_energy=100.0,
        death_penalty=50.0,
        initial_food_density=0.05,
        food_respawn_lambda=1.0,
        winter_factor=0.3,
        summer_factor=1.0,
        spring_factor=1.5,
        autumn_factor=1.2,
        max_steps=1000,
    ),
    "hard": dict(
        metabolism=1.2,
        food_value=20.0,
        start_energy=40.0,
        max_energy=100.0,
        death_penalty=50.0,
        initial_food_density=0.03,
        food_respawn_lambda=0.6,
        winter_factor=0.2,
        summer_factor=0.8,
        spring_factor=1.4,
        autumn_factor=1.0,
        max_steps=500,
    ),
    # V4 — mode evolutionary : reproduction activée, env modérément doux
    "evolve": dict(
        metabolism=0.4,
        food_value=15.0,
        start_energy=100.0,
        max_energy=150.0,
        death_penalty=5.0,
        initial_food_density=0.12,
        food_respawn_lambda=3.0,
        winter_factor=0.6,
        summer_factor=1.2,
        spring_factor=1.8,
        autumn_factor=1.3,
        max_steps=3000,
    ),
    # V5 — mode civilization : reproduction + construction activées
    "civ": dict(
        metabolism=0.4,
        food_value=15.0,
        start_energy=120.0,
        max_energy=180.0,
        death_penalty=5.0,
        initial_food_density=0.10,
        food_respawn_lambda=2.5,
        winter_factor=0.6,
        summer_factor=1.2,
        spring_factor=1.8,
        autumn_factor=1.3,
        max_steps=3000,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", type=str, default="easy", choices=list(MODE_PRESETS),
        help="Preset env (easy = agents survivent, hard = mortalite massive)",
    )
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--season-period", type=int, default=200)
    parser.add_argument(
        "--cell-px", type=int, default=18,
        help="Taille des cellules en pixels (defaut 18)",
    )
    parser.add_argument("--tick-delay-ms", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)

    # Overrides individuels (None = utiliser le preset)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-energy", type=float, default=None)
    parser.add_argument("--start-energy", type=float, default=None)
    parser.add_argument("--metabolism", type=float, default=None)
    parser.add_argument("--food-value", type=float, default=None)
    parser.add_argument("--death-penalty", type=float, default=None)
    parser.add_argument("--initial-food-density", type=float, default=None)
    parser.add_argument("--food-respawn-lambda", type=float, default=None)
    parser.add_argument("--spring-factor", type=float, default=None)
    parser.add_argument("--summer-factor", type=float, default=None)
    parser.add_argument("--autumn-factor", type=float, default=None)
    parser.add_argument("--winter-factor", type=float, default=None)
    parser.add_argument("--temp-min", type=float, default=-10.0)
    parser.add_argument("--temp-max", type=float, default=30.0)
    # V4 reproduction (activée seulement en mode 'evolve' par défaut)
    parser.add_argument(
        "--reproduction", type=str, default=None,
        choices=["on", "off"],
        help="Force reproduction on/off (default: auto, on si --mode evolve)",
    )
    parser.add_argument("--repro-threshold", type=float, default=80.0)
    parser.add_argument("--repro-cost", type=float, default=40.0)
    parser.add_argument("--repro-cooldown", type=int, default=30)
    parser.add_argument("--repro-max-pop", type=int, default=80)
    # V5 construction (activée seulement en mode 'civ' par défaut)
    parser.add_argument(
        "--build", type=str, default=None, choices=["on", "off"],
        help="Force construction on/off (default: on si --mode civ)",
    )
    parser.add_argument("--build-threshold", type=float, default=90.0)
    parser.add_argument("--build-cost", type=float, default=25.0)
    parser.add_argument("--build-rest-bonus", type=float, default=3.0)
    parser.add_argument("--build-cooldown", type=int, default=50)
    args = parser.parse_args()

    preset = MODE_PRESETS[args.mode]

    def pick(name: str, fallback_key: str | None = None):
        v = getattr(args, name.replace("-", "_"))
        if v is not None:
            return v
        return preset[fallback_key or name.replace("-", "_")]

    seasonal = SeasonalConfig(
        season_period=args.season_period,
        spring_lambda_factor=pick("spring_factor"),
        summer_lambda_factor=pick("summer_factor"),
        autumn_lambda_factor=pick("autumn_factor"),
        winter_lambda_factor=pick("winter_factor"),
        temp_min=args.temp_min,
        temp_max=args.temp_max,
    )
    # Reproduction : auto-on en mode evolve OU civ, sinon off ; --reproduction force
    if args.reproduction is None:
        repro_enabled = args.mode in ("evolve", "civ")
    else:
        repro_enabled = args.reproduction == "on"

    repro = ReproductionConfig(
        enabled=repro_enabled,
        energy_threshold=args.repro_threshold,
        energy_cost=args.repro_cost,
        cooldown_ticks=args.repro_cooldown,
        max_population=args.repro_max_pop,
    )

    # V5 construction : auto-on en mode civ uniquement, sinon off
    if args.build is None:
        build_enabled = args.mode == "civ"
    else:
        build_enabled = args.build == "on"

    build = BuildConfig(
        enabled=build_enabled,
        energy_threshold=args.build_threshold,
        build_cost=args.build_cost,
        rest_bonus=args.build_rest_bonus,
        cooldown_ticks=args.build_cooldown,
    )

    cfg = SeasonalMultiAgentConfig(
        rows=args.rows, cols=args.cols, n_agents=args.n_agents,
        max_energy=pick("max_energy"),
        start_energy=pick("start_energy"),
        metabolism=pick("metabolism"),
        food_value=pick("food_value"),
        death_penalty=pick("death_penalty"),
        initial_food_density=pick("initial_food_density"),
        food_respawn_lambda=pick("food_respawn_lambda"),
        max_steps=pick("max_steps"),
        seasonal=seasonal,
        reproduction=repro,
        build=build,
    )

    print(
        f"AetherLife V5 — mode={args.mode}  N={args.n_agents}  "
        f"grid={args.rows}x{args.cols}  period={args.season_period}\n"
        f"  metabolism={cfg.metabolism}  food_value={cfg.food_value}  "
        f"start_energy={cfg.start_energy}  max_steps={cfg.max_steps}\n"
        f"  density={cfg.initial_food_density}  respawn_lambda={cfg.food_respawn_lambda}\n"
        f"  season factors (sp/su/au/wi): "
        f"{seasonal.spring_lambda_factor}/{seasonal.summer_lambda_factor}/"
        f"{seasonal.autumn_lambda_factor}/{seasonal.winter_lambda_factor}\n"
        f"  reproduction={repro.enabled}  threshold={repro.energy_threshold}  "
        f"cost={repro.energy_cost}  cooldown={repro.cooldown_ticks}t  "
        f"max_pop={repro.max_population}\n"
        f"  construction={build.enabled}  threshold={build.energy_threshold}  "
        f"cost={build.build_cost}  rest_bonus={build.rest_bonus}  "
        f"cooldown={build.cooldown_ticks}t\n"
    )

    run_gui_v3(cfg, cell_px=args.cell_px, tick_delay_ms=args.tick_delay_ms, seed=args.seed)


if __name__ == "__main__":
    main()
