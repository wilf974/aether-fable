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
from aetherlife.world.cache import CacheConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
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
    # V5.2 — mode tribe : reproduction + construction + heritage familial
    "tribe": dict(
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
    # V5.3 — mode prosper : tribe + caches food (stockage inter-temporel)
    # V5.6 recalibré : généreux pour rendre les comportements bien visibles
    "prosper": dict(
        metabolism=0.4,
        food_value=15.0,
        start_energy=110.0,
        max_energy=180.0,
        death_penalty=5.0,
        initial_food_density=0.12,
        food_respawn_lambda=2.8,
        winter_factor=0.5,
        summer_factor=1.3,
        spring_factor=2.0,
        autumn_factor=1.3,
        max_steps=3000,
    ),
    # V6 — mode garden : prosper + plantation (système agricole complet)
    # V6.1 — food spontanée RARE : survie = cultiver ou mourir
    "garden": dict(
        metabolism=0.4,
        food_value=18.0,                  # une plante mûre rapporte gros
        start_energy=140.0,                # surplus initial pour démarrer la culture
        max_energy=220.0,
        death_penalty=5.0,
        initial_food_density=0.04,         # juste de quoi amorcer
        food_respawn_lambda=0.15,          # quasi-zéro food gratuite
        winter_factor=0.6,                 # cycle modeste (pas de famine extrême)
        summer_factor=1.0,
        spring_factor=1.4,
        autumn_factor=0.8,
        max_steps=4000,
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
    # V6.3 — seuils encore plus accessibles + cooldowns courts pour activité visible
    parser.add_argument("--repro-threshold", type=float, default=60.0)
    parser.add_argument("--repro-cost", type=float, default=30.0)
    parser.add_argument("--repro-cooldown", type=int, default=12)
    parser.add_argument("--repro-max-pop", type=int, default=60)
    # V5 construction (activée seulement en mode 'civ' par défaut)
    parser.add_argument(
        "--build", type=str, default=None, choices=["on", "off"],
        help="Force construction on/off (default: on si --mode civ)",
    )
    parser.add_argument("--build-threshold", type=float, default=55.0)
    parser.add_argument("--build-cost", type=float, default=18.0)
    parser.add_argument("--build-rest-bonus", type=float, default=4.0)
    parser.add_argument("--build-cooldown", type=int, default=15)
    parser.add_argument(
        "--family", type=lambda x: x.lower() == "on" if x else None,
        default=None,
        help="Force family inheritance on/off (default: on si --mode tribe/prosper)",
    )
    parser.add_argument(
        "--cache", type=str, default=None, choices=["on", "off"],
        help="Force cache on/off (default: on si --mode prosper)",
    )
    parser.add_argument("--cache-deposit-threshold", type=float, default=80.0)
    parser.add_argument("--cache-withdrawal-threshold", type=float, default=60.0)
    parser.add_argument("--cache-capacity", type=float, default=80.0)
    parser.add_argument("--cache-deposit-amount", type=float, default=4.0)
    parser.add_argument("--cache-withdrawal-amount", type=float, default=4.0)
    # V6 plantation (activée seulement en mode 'garden' par défaut)
    # V6.3 — seuils accessibles + plus de graines initiales
    parser.add_argument(
        "--planting", type=str, default=None, choices=["on", "off"],
        help="Force plantation on/off (default: on si --mode garden)",
    )
    parser.add_argument("--plant-threshold", type=float, default=65.0)
    parser.add_argument("--plant-cost", type=float, default=10.0)
    parser.add_argument("--plant-growth-ticks", type=int, default=50)
    parser.add_argument("--plant-cooldown", type=int, default=10)
    parser.add_argument("--initial-seeds", type=int, default=3)
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

    # V5 construction : auto-on en mode civ/tribe/prosper, sinon off
    if args.build is None:
        build_enabled = args.mode in ("civ", "tribe", "prosper")
    else:
        build_enabled = args.build == "on"

    # V5.2 family inheritance : on automatiquement en mode tribe ou prosper
    family_inheritance = (
        args.family if args.family is not None
        else args.mode in ("tribe", "prosper")
    )

    # V5.3 caches : auto-on en mode prosper uniquement
    if args.cache is None:
        cache_enabled = args.mode == "prosper"
    else:
        cache_enabled = args.cache == "on"

    build = BuildConfig(
        enabled=build_enabled,
        energy_threshold=args.build_threshold,
        build_cost=args.build_cost,
        rest_bonus=args.build_rest_bonus,
        cooldown_ticks=args.build_cooldown,
        family_inheritance=family_inheritance,
    )

    cache = CacheConfig(
        enabled=cache_enabled,
        deposit_threshold=args.cache_deposit_threshold,
        withdrawal_threshold=args.cache_withdrawal_threshold,
        max_capacity=args.cache_capacity,
        deposit_amount=args.cache_deposit_amount,
        withdrawal_amount=args.cache_withdrawal_amount,
    )

    # V6 plantation : auto-on en mode garden uniquement
    if args.planting is None:
        plant_enabled = args.mode == "garden"
    else:
        plant_enabled = args.planting == "on"

    planting = PlantingConfig(
        enabled=plant_enabled,
        energy_threshold=args.plant_threshold,
        energy_cost=args.plant_cost,
        growth_ticks=args.plant_growth_ticks,
        cooldown_ticks=args.plant_cooldown,
        seeds_required=1,
        seeds_per_food_eaten=1,
        initial_seeds=args.initial_seeds,
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
        cache=cache,
        planting=planting,
    )

    print(
        f"AetherLife V5.3 — mode={args.mode}  N={args.n_agents}  "
        f"grid={args.rows}x{args.cols}  period={args.season_period}\n"
        f"  metabolism={cfg.metabolism}  food_value={cfg.food_value}  "
        f"start_energy={cfg.start_energy}  max_steps={cfg.max_steps}\n"
        f"  density={cfg.initial_food_density}  respawn_lambda={cfg.food_respawn_lambda}\n"
        f"  season factors (sp/su/au/wi): "
        f"{seasonal.spring_lambda_factor}/{seasonal.summer_lambda_factor}/"
        f"{seasonal.autumn_lambda_factor}/{seasonal.winter_lambda_factor}\n"
        f"  reproduction={repro.enabled}  build={build.enabled}  "
        f"family_inheritance={build.family_inheritance}\n"
        f"  cache={cache.enabled}  deposit>={cache.deposit_threshold}  "
        f"withdraw<{cache.withdrawal_threshold}  cap={cache.max_capacity}\n"
        f"  planting={planting.enabled}  threshold={planting.energy_threshold}  "
        f"cost={planting.energy_cost}  growth={planting.growth_ticks}t\n"
    )

    run_gui_v3(cfg, cell_px=args.cell_px, tick_delay_ms=args.tick_delay_ms, seed=args.seed)


if __name__ == "__main__":
    main()
