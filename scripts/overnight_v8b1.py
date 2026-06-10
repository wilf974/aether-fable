"""V8-B1 overnight validation — 50k+ ticks pour valider les 4 critères.

Critères de succès V8-B1 (avis user 2026-05-23) :

  1. HÉRITAGE COGNITIF RÉEL
     Mesure : lifespan moyen monte avec birth_tick. Si un agent né à
     t=40000 vit plus longtemps qu'un fondateur (t=0), c'est parce
     qu'il a hérité d'un meilleur cerveau, pas par chance.

  2. DIVERGENCE DE LIGNÉES
     Mesure : pour chaque paire de lignées vivantes, distribution des
     actions sur observation identique. Si KL > seuil, divergence
     stratégique. Si KL ≈ 0, toutes lignées convergent vers même
     politique (mauvais signe).

  3. SÉLECTION COGNITIVE OBSERVABLE
     Mesure : courbe n_lineages_alive(t). Si elle baisse monotonement
     puis se stabilise, c'est de la sélection. Si stable d'emblée, pas
     de pression.

  4. MÉMOIRE INTERGÉNÉRATIONNELLE
     Mesure : compare gen 0 vs gen N en isolation. Spawn 5 agents avec
     poids gen 0 (savé au début) et 5 agents avec poids final. Le
     deuxième groupe doit survivre plus longtemps.

Output : log JSON détaillé toutes les 5k ticks + rapport final ASCII.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from aetherlife.agents.lineage_agent import LineageAgent, egocentric_obs
from aetherlife.agents.lineage_brain import BrainConfig, LineageBrain
from aetherlife.telemetry import MetricsLogger
from aetherlife.world.biomes import BiomeConfig
from aetherlife.world.cache import CacheConfig
from aetherlife.world.competition import CompetitionConfig
from aetherlife.world.cooperative import CooperativeConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.language_causality import LanguageCausalityTracker
from aetherlife.world.seasonal_grid import (
    SeasonalConfig, SeasonalMultiAgentConfig, SeasonalMultiAgentFoodGrid,
)
from aetherlife.world.vocabulary import VocabularyConfig
from aetherlife.historian.spatial_mobility import (
    OccupancyAccumulator, build_spatial_mobility_block, window_bounds,
)


def build_env(
    seed: int, *, regime: str = "training",
    disable_vocalize_after_tick: int | None = None,
    vocalize_energy_cost: float = 0.05,
    max_pop_override: int | None = None,
    bonus_energy_override: float | None = None,
    n_initial_affinities: int = 4,
    n_seed_points: int = 8,
) -> SeasonalMultiAgentFoodGrid:
    """Config selon regime :

    - 'training' : permissif, monde homogène (V8-B1).
    - 'darwinian' : dur, monde homogène (V8-B1).
    - 'niches'   : V8-B1.5 — biomes Voronoi + compétition locale +
                   max_pop élevé. Cible : ≥3 lignées coexistantes après 100k.
    """
    _COORD_REGIMES = {
        "coordination", "coordination_hidden",
        "coordination_hard", "coordination_collective",
    }
    if n_initial_affinities != 4 and regime not in _COORD_REGIMES:
        raise ValueError(
            f"n_initial_affinities={n_initial_affinities} non supporte hors "
            f"regime coordination (got regime={regime!r})"
        )
    if regime == "darwinian":
        metabolism = 0.65
        winter_f = 0.3
        start_e = 140.0
    else:
        metabolism = 0.35
        winter_f = 0.5
        start_e = 160.0
    # V8-B1.5 : régime "niches" → biomes + compétition + max_pop haut
    if regime == "niches":
        rows = 40
        cols = 40
        n_agents = 16
        max_pop = 80
        food_respawn_lambda = 0.6
        metabolism = 0.3
        start_e = 180.0
        biome_cfg = BiomeConfig(enabled=True, n_seed_points=6)
        competition_cfg = CompetitionConfig(
            enabled=True, radius=3,
            metabolism_per_neighbor=0.03, max_factor=2.0,
        )
    elif regime == "speciation":
        # V8-B1.6 + V8-B1.7 : biomes + affinity héritée + reproduction
        # biome-locked + seed bank + respawn anti-extinction
        rows = 40
        cols = 40
        n_agents = 20    # 5 par affinity en moyenne
        max_pop = 100
        food_respawn_lambda = 0.6
        metabolism = 0.3
        start_e = 180.0
        biome_cfg = BiomeConfig(
            enabled=True, n_seed_points=8, balanced_seeds=True,
            affinity_enabled=True,
            in_affinity_metabolism=0.7, in_affinity_food_value=1.3,
            out_affinity_metabolism=1.5, out_affinity_food_value=0.7,
            out_affinity_movement_mult=2.5,
            reproduction_locked_to_affinity=True,
            respawn_enabled=True,
            respawn_check_every=200,
            respawn_extinct_after_ticks=3000,
            respawn_threshold=2,
            respawn_initial_energy=200.0,
            seed_bank_max_per_affinity=2,
            seed_bank_mutation_std=0.05,
        )
        competition_cfg = CompetitionConfig(
            enabled=True, radius=3,
            metabolism_per_neighbor=0.03, max_factor=2.0,
        )
    elif regime == "language":
        # V8-B2.0 : speciation + vocabulary émergent (pas de reward direct)
        rows = 40
        cols = 40
        n_agents = 20
        max_pop = 100
        food_respawn_lambda = 0.6
        metabolism = 0.3
        start_e = 180.0
        biome_cfg = BiomeConfig(
            enabled=True, n_seed_points=8, balanced_seeds=True,
            affinity_enabled=True,
            in_affinity_metabolism=0.7, in_affinity_food_value=1.3,
            out_affinity_metabolism=1.5, out_affinity_food_value=0.7,
            out_affinity_movement_mult=2.5,
            reproduction_locked_to_affinity=True,
            respawn_enabled=True,
            respawn_check_every=200,
            respawn_extinct_after_ticks=3000,
            respawn_threshold=2,
            respawn_initial_energy=200.0,
            seed_bank_max_per_affinity=2,
        )
        competition_cfg = CompetitionConfig(
            enabled=True, radius=3,
            metabolism_per_neighbor=0.03, max_factor=2.0,
        )
    elif regime in (
        "coordination", "coordination_hidden", "coordination_hard",
        "coordination_collective",
    ):
        # V8-C1 : ressources invisibles éloignées (vision=2, listen=10)
        # V8-C2 : V8-C1 + food invisible (hidden_food=True dans BiomeConfig)
        # V8-C2.b : V8-C2 + ECO DUR (max_pop=50, winter=0.3, respawn=1)
        #          pour faire émerger la pression où la communication compte.
        rows = 40
        cols = 40
        n_agents = 20
        if regime == "coordination_hard":
            # V8-C2.b'' : minimal hardening.
            max_pop = 80
            food_respawn_lambda = 0.55
            metabolism = 0.3
            winter_f = 0.5
            respawn_thr = 2
            start_e = 200.0
        elif regime == "coordination_collective":
            # V8-C3a'' (curriculum raffiné après diagnostic mono-density) :
            # Smoke C3a' 15k → 0/4 patterns positifs car densité (mean=23
            # voisins/r3) rendait la coop triviale par hasard fréquent.
            # On baisse max_pop=60 pour FORCER les agents à se chercher
            # activement → coop SÉLECTIONNÉE, pas opportuniste.
            max_pop = 60
            food_respawn_lambda = 0.6
            metabolism = 0.3
            winter_f = 0.5
            respawn_thr = 2
            start_e = 220.0
        else:
            max_pop = 100
            food_respawn_lambda = 0.6
            metabolism = 0.3
            winter_f = 0.5
            respawn_thr = 2
            start_e = 200.0
        biome_cfg = BiomeConfig(
            enabled=True, n_seed_points=n_seed_points, balanced_seeds=True,
            affinity_enabled=True,
            in_affinity_metabolism=0.7, in_affinity_food_value=1.3,
            out_affinity_metabolism=1.5, out_affinity_food_value=0.7,
            out_affinity_movement_mult=2.5,
            reproduction_locked_to_affinity=True,
            respawn_enabled=True,
            respawn_check_every=200,
            respawn_extinct_after_ticks=3000,
            respawn_threshold=respawn_thr,
            respawn_initial_energy=200.0,
            seed_bank_max_per_affinity=2,
            # V8-C2/V8-C2.b : food invisible à la vue.
            # V8-C3a (curriculum init) : food VISIBLE pour apprendre gather
            # sans cumuler la difficulté hidden_food. C3b durcira plus tard.
            hidden_food=(regime in (
                "coordination_hidden", "coordination_hard",
            )),
            n_initial_affinities=n_initial_affinities,
        )
        competition_cfg = CompetitionConfig(
            enabled=True, radius=3,
            metabolism_per_neighbor=0.03, max_factor=2.0,
        )
    else:
        rows = 30
        cols = 30
        n_agents = 10
        max_pop = 30
        food_respawn_lambda = 0.25
        biome_cfg = BiomeConfig(enabled=False)
        competition_cfg = CompetitionConfig(enabled=False)
    # V8-B2.0 — vocabulary activée seulement en mode language/coordination
    if regime == "language":
        vocab_cfg = VocabularyConfig(
            enabled=True, n_tokens=4, embedding_dim=16,
            listen_radius=5, mutation_std=0.05,
            vocalize_energy_cost=vocalize_energy_cost, social_bonus=0.0,
            disable_vocalize_after_tick=disable_vocalize_after_tick,
        )
    elif regime in (
        "coordination", "coordination_hidden", "coordination_hard",
        "coordination_collective",
    ):
        # V8-C1/C2/C2.b/C3 : listen_radius=10 (vs vision_radius=2)
        vocab_cfg = VocabularyConfig(
            enabled=True, n_tokens=4, embedding_dim=16,
            listen_radius=10, mutation_std=0.05,
            vocalize_energy_cost=vocalize_energy_cost, social_bonus=0.0,
            disable_vocalize_after_tick=disable_vocalize_after_tick,
        )
    else:
        vocab_cfg = VocabularyConfig(enabled=False)
    # V8-C3 — actions coopératives lourdes (gather_collective)
    if regime == "coordination_collective":
        # V8-C3a''-soft : isoler le levier "densité" (diagnostic smoke C3a').
        # C3a'' avec min_partners=2 → extinction t=1281 (combo trop dur).
        # Nouvelle approche : SEUL levier modifié vs C3a' = max_pop 100→60
        # (couplé côté env). On garde min_partners=1 (coop duo) pour
        # ne pas cumuler les difficultés. Hypothèse : baisser la densité
        # naturelle suffit à passer "coop par hasard" → "coop sélectionnée".
        # V8-C3 P1 : overrides CLI pour tester P1 (curriculum optimisé).
        bonus_e = (
            bonus_energy_override
            if bonus_energy_override is not None else 100.0
        )
        cooperative_cfg = CooperativeConfig(
            enabled=True,
            min_partners_adjacent=1,
            signal_window_ticks=5,
            bonus_energy=bonus_e,
            spawn_lambda=0.5,
            decay_ticks=100,
            max_active_spots=40,
        )
    else:
        cooperative_cfg = CooperativeConfig(enabled=False)
    # V8-C3 P1 : override max_pop si fourni (test curriculum optimisé)
    if max_pop_override is not None and regime == "coordination_collective":
        max_pop = max_pop_override
    cfg = SeasonalMultiAgentConfig(
        rows=rows, cols=cols, n_agents=n_agents,
        max_energy=300.0, start_energy=start_e,
        metabolism=metabolism, food_value=18.0, death_penalty=0.0,
        initial_food_density=0.06,
        food_respawn_lambda=food_respawn_lambda,
        max_steps=200_000,
        seasonal=SeasonalConfig(
            season_period=200, spring_lambda_factor=1.4,
            summer_lambda_factor=1.0, autumn_lambda_factor=0.8,
            winter_lambda_factor=winter_f,
        ),
        reproduction=ReproductionConfig(
            enabled=True, energy_threshold=130.0, energy_cost=70.0,
            cooldown_ticks=100, max_population=max_pop,
        ),
        build=BuildConfig(
            enabled=True, energy_threshold=130.0, build_cost=40.0,
            rest_bonus=4.0, cooldown_ticks=100, family_inheritance=True,
        ),
        cache=CacheConfig(enabled=False),
        planting=PlantingConfig(enabled=False),
        biomes=biome_cfg,
        competition=competition_cfg,
        vocabulary=vocab_cfg,
        cooperative=cooperative_cfg,
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=seed)
    return env


def policy_distribution(brain: LineageBrain, obs_samples: np.ndarray) -> np.ndarray:
    """Calcule la distribution d'actions du brain sur un set d'obs identiques.

    Returns p ∈ R^n_actions avec p_a = freq(argmax Q(s, .))
    """
    torch = brain._torch  # noqa: SLF001
    counts = np.zeros(brain.n_actions, dtype=np.float32)
    with torch.no_grad():
        x = torch.from_numpy(obs_samples).float().to(brain.device)
        q = brain.online(x)
        actions = q.argmax(dim=1).cpu().numpy()
        for a in actions:
            counts[a] += 1
    counts /= max(counts.sum(), 1)
    return counts


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def measure_policy_divergence(policy: LineageAgent, n_samples: int = 32) -> float:
    """Pour chaque paire de lignées vivantes, calcule KL des distributions
    d'actions sur des obs identiques (random gaussien). Retourne moyenne."""
    brains = list(policy.registry)
    if len(brains) < 2:
        return 0.0
    # Génère un mini-batch d'obs identiques (random) que tous les brains voient
    obs_batch = policy._fresh_seed()  # noqa: SLF001
    rng = np.random.default_rng(obs_batch)
    obs = rng.standard_normal((n_samples, policy.obs_dim)).astype(np.float32)
    dists = [policy_distribution(b, obs) for b in brains]
    kls: list[float] = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            kls.append(kl_divergence(dists[i], dists[j]))
            kls.append(kl_divergence(dists[j], dists[i]))
    return float(np.mean(kls)) if kls else 0.0


def lifespan_by_birth_quartile(deaths: list[tuple[int, int]]) -> dict:
    """deaths = list of (birth_tick, death_tick). Group by quartile."""
    if not deaths:
        return {}
    births = sorted(b for b, _ in deaths)
    q1 = births[len(births) // 4] if len(births) >= 4 else births[0]
    q2 = births[len(births) // 2]
    q3 = births[3 * len(births) // 4] if len(births) >= 4 else births[-1]
    buckets = defaultdict(list)
    for b, d in deaths:
        lifespan = d - b
        if b <= q1:
            buckets["Q1_early"].append(lifespan)
        elif b <= q2:
            buckets["Q2"].append(lifespan)
        elif b <= q3:
            buckets["Q3"].append(lifespan)
        else:
            buckets["Q4_late"].append(lifespan)
    return {
        k: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v))}
        for k, v in buckets.items()
    }


def run_overnight(
    n_ticks: int, seed: int, device: str, out_dir: str,
    snap_every: int = 5000, divergence_every: int = 5000,
    regime: str = "training",
    disable_vocalize_after_tick: int | None = None,
    vocalize_energy_cost: float = 0.05,
    max_pop_override: int | None = None,
    bonus_energy_override: float | None = None,
    n_initial_affinities: int = 4,
    n_seed_points: int = 8,
) -> dict:
    env = build_env(
        seed, regime=regime,
        disable_vocalize_after_tick=disable_vocalize_after_tick,
        vocalize_energy_cost=vocalize_energy_cost,
        max_pop_override=max_pop_override,
        bonus_energy_override=bonus_energy_override,
        n_initial_affinities=n_initial_affinities,
        n_seed_points=n_seed_points,
    )
    print(f"REGIME={regime}"
          + (f"  ABLATION@{disable_vocalize_after_tick}"
             if disable_vocalize_after_tick is not None else "")
          + (f"  VCOST={vocalize_energy_cost}"
             if vocalize_energy_cost != 0.05 else ""))
    # V8-B2.1 — re-stabilisation pour action space étendu (8 actions
    # avec langage). lr plus bas, target sync plus fréquent, epsilon_end
    # légèrement >0 pour exploration résiduelle.
    # V8-C1/C2/C2.b : vision réduite (2 cases) en régimes coordination
    vision_radius = 2 if regime in (
        "coordination", "coordination_hidden", "coordination_hard",
    ) else 4
    cfg = BrainConfig(
        enabled=True, device=device, vision_radius=vision_radius,
        hidden_dims=(64, 64), lr=1e-4, batch_size=64,
        buffer_capacity=50_000, min_replay_to_learn=500, train_every=4,
        epsilon_start=0.6, epsilon_end=0.08, epsilon_decay_steps=30_000,
        target_sync_steps=200, mutation_std=0.03,
    )
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=seed)
    print(f"OBS_DIM={policy.obs_dim}  device={cfg.device}  brains={len(policy.registry)}")

    # V8-B2.2 — Causality tracker (observer pur, jamais d'influence)
    causality_tracker: LanguageCausalityTracker | None = None
    if env.cfg.vocabulary.enabled:
        causality_tracker = LanguageCausalityTracker(
            n_tokens=env.cfg.vocabulary.n_tokens,
            n_actions=policy.n_actions,
            post_listen_window=3,
        )

    last_ego: dict[int, np.ndarray] = {
        a.agent_id: policy.make_obs(a)
        for a in env._agents  # noqa: SLF001
        if a.alive
    }

    # Tracking
    deaths_record: list[tuple[int, int]] = []  # (birth_tick, death_tick)
    alive_curve: list[tuple[int, int]] = []
    lineages_curve: list[tuple[int, int]] = []
    divergence_curve: list[tuple[int, float]] = []
    loss_curve: list[tuple[int, float]] = []
    last_seen_alive: set[int] = {a.agent_id for a in env._agents if a.alive}  # noqa: SLF001

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # V2.5 — telemetrie structuree : metrics.jsonl + run_config.json
    # (observation pure, aucun impact dynamique/RNG)
    mlog = MetricsLogger(
        out_dir,
        run_id=f"{regime}_seed{seed}",
        config={
            "n_ticks": n_ticks, "seed": seed, "device": device,
            "regime": regime, "n_initial_affinities": n_initial_affinities,
            "n_seed_points": n_seed_points,
            "vocalize_energy_cost": vocalize_energy_cost,
            "disable_vocalize_after_tick": disable_vocalize_after_tick,
            "brain": {"vision_radius": cfg.vision_radius, "lr": cfg.lr,
                       "hidden_dims": list(cfg.hidden_dims)},
        },
    )

    # V8-C3 chantier A — rétention occupation (observation-only : aucun impact
    # sur la dynamique/RNG). Fenêtres officielles = 1er tiers vs 3e tiers
    # (exclut le transitoire de fondation — cf. window_bounds).
    _mob_swin, _mob_ewin = window_bounds(n_ticks)
    _mob_w = _mob_swin[1]  # = n_ticks // 3
    _mob_start = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    _mob_end = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)

    for t in range(1, n_ticks + 1):
        if env.n_alive == 0:
            print(f"[t={t}] EXTINCTION")
            break

        obs_stub = {a.agent_id: np.zeros(10) for a in env._agents if a.alive}  # noqa: SLF001
        actions = policy.act_dict(obs_stub, greedy=False)
        e_before = {
            a.agent_id: a.energy for a in env._agents if a.alive  # noqa: SLF001
        }
        # V8-B2.2 — Push actions au tracker AVANT step (état pré-action)
        if causality_tracker is not None:
            causality_tracker.push_actions(t, dict(actions))
        env.step(actions)

        # V8-C3 chantier A — accumulation occupation (fenêtres début/fin)
        if t <= _mob_w or t > n_ticks - _mob_w:
            _pos = [
                (a.pos[0], a.pos[1])
                for a in env._agents if a.alive  # noqa: SLF001
            ]
            (_mob_start if t <= _mob_w else _mob_end).add_positions(_pos)

        # V8-B2.2 — Push emissions de tokens (lecture du buffer env._tokens_this_tick)
        if causality_tracker is not None and env.cfg.vocabulary.enabled:
            tokens_this_tick = getattr(env, "_tokens_this_tick", {})  # noqa: SLF001
            listen_r = env.cfg.vocabulary.listen_radius
            for speaker_id, token_id in tokens_this_tick.items():
                try:
                    speaker = env.agent_state(speaker_id)
                except KeyError:
                    continue
                if not speaker.alive:
                    continue
                # Identifier les listeners dans listen_radius
                listener_ids: list[int] = []
                for other in env._agents:  # noqa: SLF001
                    if not other.alive or other.agent_id == speaker_id:
                        continue
                    d = (abs(other.pos[0] - speaker.pos[0])
                         + abs(other.pos[1] - speaker.pos[1]))
                    if d <= listen_r:
                        listener_ids.append(other.agent_id)
                # Capturer le contexte d'émission (read-only)
                food_visible = False
                # Check food dans listen_r autour du speaker
                for r in range(
                    max(0, speaker.pos[0] - listen_r),
                    min(env.cfg.rows, speaker.pos[0] + listen_r + 1),
                ):
                    for c in range(
                        max(0, speaker.pos[1] - listen_r),
                        min(env.cfg.cols, speaker.pos[1] + listen_r + 1),
                    ):
                        if env.food_mask[r, c]:
                            food_visible = True
                            break
                    if food_visible:
                        break
                ctx = {
                    "energy": float(speaker.energy / env.cfg.max_energy),
                    "food_visible": food_visible,
                    "n_neighbors": float(min(len(listener_ids), 5) / 5.0),
                    "biome": int(env._biome_map[speaker.pos[0], speaker.pos[1]])  # noqa: SLF001
                        if env.cfg.biomes.enabled else 0,
                }
                causality_tracker.push_emission(
                    tick=t, speaker_id=speaker_id, token_id=token_id,
                    listener_ids=listener_ids, context=ctx,
                )

        # Construct transitions
        next_ego: dict[int, np.ndarray] = {}
        rewards: dict[int, float] = {}
        dones: dict[int, bool] = {}
        roots_now: dict[int, int] = {}
        for a in env._agents:  # noqa: SLF001
            if a.agent_id not in e_before:
                continue
            roots_now[a.agent_id] = a.root_ancestor_id
            if a.alive:
                next_ego[a.agent_id] = policy.make_obs(a)
                rewards[a.agent_id] = (a.energy - e_before[a.agent_id]) * 0.1
                dones[a.agent_id] = False
            else:
                next_ego[a.agent_id] = last_ego.get(
                    a.agent_id, np.zeros(policy.obs_dim, dtype=np.float32),
                )
                rewards[a.agent_id] = -5.0
                dones[a.agent_id] = True

        metrics = policy.observe_dict(
            prev_obs_ego=last_ego, actions=actions, rewards=rewards,
            next_obs_ego=next_ego, dones=dones, agent_root_ids=roots_now,
        )

        # V8-B1.7 — respawn extinct affinities via seed bank
        policy.maybe_respawn_extinct_affinities()

        # Detect deaths
        cur_alive = {a.agent_id for a in env._agents if a.alive}  # noqa: SLF001
        for did in last_seen_alive - cur_alive:
            try:
                ag = env.agent_state(did)
                deaths_record.append((ag.birth_tick, t))
            except KeyError:
                pass
        last_seen_alive = cur_alive

        # Update last_ego pour next tick
        for a in env._agents:  # noqa: SLF001
            if a.alive and a.agent_id not in last_ego:
                last_ego[a.agent_id] = policy.make_obs(a)
        last_ego = next_ego

        # Snapshots periodiques
        if t % snap_every == 0:
            elapsed = time.time() - t0
            speed = t / elapsed
            mean_loss = metrics.get("mean_loss", float("nan"))
            print(
                f"[t={t:6d}] alive={env.n_alive:3d} lineages={len(policy.registry):3d} "
                f"births={env.n_births_total:4d} steps_total={policy.registry.total_global_steps():7d} "
                f"loss={mean_loss:.4f} eps={metrics.get('mean_epsilon', 0):.3f} "
                f"speed={speed:.0f} t/s ETA={(n_ticks - t) / speed / 60:.1f}min"
            )
            alive_curve.append((t, env.n_alive))
            mlog.log(
                t, alive=env.n_alive, lineages=len(policy.registry),
                births=env.n_births_total,
                steps_total=policy.registry.total_global_steps(),
                mean_loss=None if np.isnan(mean_loss) else float(mean_loss),
                mean_epsilon=float(metrics.get("mean_epsilon", 0.0)),
                ticks_per_s=round(speed, 1),
            )
            lineages_curve.append((t, len(policy.registry)))
            if not np.isnan(mean_loss):
                loss_curve.append((t, mean_loss))

            # Cull periodic
            alive_roots = {a.root_ancestor_id for a in env._agents if a.alive}  # noqa: SLF001
            policy.registry.cull_dead_lineages(alive_roots)

        if t % divergence_every == 0:
            div = measure_policy_divergence(policy, n_samples=64)
            divergence_curve.append((t, div))
            mlog.log(t, policy_divergence=float(div))

    dt = time.time() - t0
    mlog.summary(
        final_alive=env.n_alive, births_total=env.n_births_total,
        deaths_total=len(deaths_record), ticks_done=t,
    )
    mlog.close()

    # Final analysis
    lifespan_quartiles = lifespan_by_birth_quartile(deaths_record)
    final_divergence = measure_policy_divergence(policy, n_samples=128)

    final_lineage_counts = Counter(
        a.root_ancestor_id for a in env._agents if a.alive  # noqa: SLF001
    )
    dom = final_lineage_counts.most_common(3)
    dominant_pct = (
        100 * dom[0][1] / max(env.n_alive, 1)
        if dom else 0.0
    )
    # V8-B1.6 : distribution des affinities vivantes
    affinity_counts = Counter(
        a.biome_affinity for a in env._agents  # noqa: SLF001
        if a.alive and a.biome_affinity is not None
    )

    # V8-B2.0 — métriques d'émergence du langage (PAS de traduction sémantique,
    # uniquement observations probabilistes)
    language_metrics: dict = {}
    if env.cfg.vocabulary.enabled:
        brains_with_vocab = [b for b in policy.registry if b.vocabulary is not None]
        if brains_with_vocab:
            total_tokens = sum(
                int(b.vocabulary.usage_count.sum()) for b in brains_with_vocab
            )
            # tokens_per_1000_ticks
            tokens_per_1000 = (
                1000 * total_tokens / max(n_ticks, 1)
            )
            # vocalize_energy_cost_total (estimation)
            energy_total = total_tokens * env.cfg.vocabulary.vocalize_energy_cost
            # token_lineage_concentration : pour chaque token, % d'usage
            # concentré dans une seule lignée
            n_tokens = env.cfg.vocabulary.n_tokens
            concentrations = []
            for tok in range(n_tokens):
                per_lineage = {
                    b.root_id: int(b.vocabulary.usage_count[tok])
                    for b in brains_with_vocab
                }
                tot = sum(per_lineage.values())
                if tot > 0:
                    max_share = max(per_lineage.values()) / tot
                    concentrations.append(max_share)
            mean_concentration = (
                float(np.mean(concentrations)) if concentrations else 0.0
            )
            # entropies usage par lignée
            entropies = [b.vocabulary.usage_entropy() for b in brains_with_vocab]
            mean_entropy = float(np.mean(entropies)) if entropies else 0.0
            # divergence linguistique inter-lignées (L2 entre vocabularies)
            distances = []
            for i, b1 in enumerate(brains_with_vocab):
                for b2 in brains_with_vocab[i + 1:]:
                    distances.append(b1.vocabulary.distance_to(b2.vocabulary))
            mean_distance = float(np.mean(distances)) if distances else 0.0
            language_metrics = {
                "n_brains_with_vocab": len(brains_with_vocab),
                "total_vocalize_count": total_tokens,
                "tokens_per_1000_ticks": tokens_per_1000,
                "vocalize_energy_cost_total": energy_total,
                "mean_token_lineage_concentration": mean_concentration,
                "mean_usage_entropy": mean_entropy,
                "max_possible_entropy": float(np.log(n_tokens)),
                "entropy_ratio": mean_entropy / max(float(np.log(n_tokens)), 1e-9),
                "mean_inter_lineage_distance": mean_distance,
                "per_token_usage_top": {
                    str(t): sum(
                        int(b.vocabulary.usage_count[t])
                        for b in brains_with_vocab
                    )
                    for t in range(n_tokens)
                },
            }

    final_report = {
        "config": {
            "n_ticks": n_ticks, "seed": seed, "device": device,
            "obs_dim": policy.obs_dim, "vision_radius": cfg.vision_radius,
            "n_initial_affinities": n_initial_affinities,
            "n_seed_points": n_seed_points,
        },
        "runtime": {
            "duration_s": dt, "ticks_per_sec": n_ticks / dt,
        },
        "criterion_1_inheritance": {
            "lifespan_by_birth_quartile": lifespan_quartiles,
            "interpretation": (
                "Si Q4_late.mean > Q1_early.mean → héritage cognitif positif"
                if all(k in lifespan_quartiles for k in ("Q1_early", "Q4_late"))
                else "données insuffisantes"
            ),
        },
        "criterion_2_divergence": {
            "final_kl_mean": final_divergence,
            "divergence_curve_sample": divergence_curve[-10:],
            "interpretation": (
                "KL > 0.1 = lignées divergent stratégiquement"
                if final_divergence > 0.1
                else "KL faible : politiques homogènes — mauvais signe"
            ),
        },
        "criterion_3_selection": {
            "n_lineages_initial": 10,
            "n_lineages_final": len(final_lineage_counts),
            "n_lineages_curve_sample": lineages_curve[-10:],
            "dominant_lineage_pct": dominant_pct,
            "interpretation": (
                f"Sélection visible : {10 - len(final_lineage_counts)} "
                f"lignées éteintes sur 10"
            ),
        },
        "criterion_4_memory": {
            "total_brain_steps": policy.registry.total_global_steps(),
            "note": "Test gen0 vs genN à implémenter en post-analyse",
        },
        "final_state": {
            "n_alive": env.n_alive,
            "n_births_total": env.n_births_total,
            "n_deaths": len(deaths_record),
            "top_lineages": [
                {
                    "root_id": r, "alive": n,
                    "pct": 100 * n / max(env.n_alive, 1),
                }
                for r, n in dom
            ],
            "affinity_distribution": {
                str(k): v for k, v in affinity_counts.items()
            },
            "n_affinities_alive": len(affinity_counts),
        },
        "language_metrics_v8b2": language_metrics,
        "language_causality_v8b2_2": (
            causality_tracker.finalize() if causality_tracker is not None else {}
        ),
        "cooperative_v8c3": {
            "enabled": bool(env.cfg.cooperative.enabled),
            "gather_successes_total": int(env.gather_successes_total),
            "gather_failures_total": int(env.gather_failures_total),
            "gather_success_rate": (
                env.gather_successes_total
                / max(env.gather_successes_total + env.gather_failures_total, 1)
            ),
            "active_spots_final": len(env.gather_spots),
            "bonus_energy": env.cfg.cooperative.bonus_energy,
            "spawn_lambda": env.cfg.cooperative.spawn_lambda,
            "decay_ticks": env.cfg.cooperative.decay_ticks,
        },
        "cooperative_metrics_v8c3": (
            env.coop_metrics.finalize()
            if env.cfg.cooperative.enabled else {}
        ),
        "curves": {
            "alive": alive_curve,
            "lineages": lineages_curve,
            "divergence": divergence_curve,
            "loss": loss_curve,
        },
        "spatial_mobility_v8c3": build_spatial_mobility_block(
            _mob_start, _mob_end,
            start_window=_mob_swin,
            end_window=_mob_ewin,
        ),
    }

    report_path = os.path.join(out_dir, f"overnight_v8b1_seed{seed}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f"\nReport saved : {report_path}")

    # Historian — observer/reporter SANS influence sur les agents.
    # Si import échoue ou erreur, on continue (fail-safe).
    try:
        from aetherlife.historian import Historian
        run_id = f"seed{seed}_t{n_ticks}_{regime}"
        historian = Historian.from_report(final_report, run_id=run_id)
        report_subdir = os.path.join(out_dir, "report")
        files = historian.write_all(report_subdir)
        print(f"Historian wrote {len(files)} files to {report_subdir}")
    except Exception as e:
        print(f"Historian skipped (fail-safe) : {e}")

    print("\n" + "=" * 60)
    print("RAPPORT FINAL V8-B1 — Validation 4 critères")
    print("=" * 60)
    print(f"\nDurée : {dt/60:.1f} min  ({n_ticks / dt:.0f} ticks/s)")
    print(f"État final : {env.n_alive} vivants, {env.n_births_total} naissances, "
          f"{len(deaths_record)} morts")
    print(f"\n--- CRITÈRE 1 : Héritage cognitif (lifespan par quartile) ---")
    for q in ("Q1_early", "Q2", "Q3", "Q4_late"):
        if q in lifespan_quartiles:
            v = lifespan_quartiles[q]
            print(f"  {q}: n={v['n']:3d}  mean_lifespan={v['mean']:6.1f}  median={v['median']:6.0f}")
    if "Q1_early" in lifespan_quartiles and "Q4_late" in lifespan_quartiles:
        delta = (
            lifespan_quartiles["Q4_late"]["mean"]
            - lifespan_quartiles["Q1_early"]["mean"]
        )
        verdict = "OK (héritage positif)" if delta > 0 else "NEGATIF (sélection ne marche pas)"
        print(f"  Delta Q4-Q1 = {delta:+.1f}  => {verdict}")

    print(f"\n--- CRITÈRE 2 : Divergence cognitive ---")
    print(f"  KL moyen inter-lignées (fin) : {final_divergence:.4f}")
    print(f"  Verdict : {'CONVERGENT (mauvais)' if final_divergence < 0.05 else 'DIVERGENT (OK)'}")

    print(f"\n--- CRITÈRE 3 : Sélection cognitive ---")
    print(f"  Lignées init : 10  →  final : {len(final_lineage_counts)}")
    print(f"  Dominance top : {dominant_pct:.1f}%")
    print(f"  Verdict : {'sélection observable' if len(final_lineage_counts) < 10 else 'pas d eradication visible'}")

    print(f"\n--- CRITÈRE 4 : Mémoire intergen ---")
    print(f"  Total brain_steps cumulés : {policy.registry.total_global_steps()}")
    print(f"  (test isolé gen0 vs genN à compléter en analyse post)")

    # V8-B2.0 — Métriques émergence du langage
    if language_metrics:
        print(f"\n--- LANGUAGE EMERGENCE V8-B2 (observations probabilistes) ---")
        print(f"  Total vocalize : {language_metrics['total_vocalize_count']}")
        print(f"  Tokens / 1000 ticks : {language_metrics['tokens_per_1000_ticks']:.1f}")
        print(f"  Energy cost total (vocalize) : {language_metrics['vocalize_energy_cost_total']:.1f}")
        print(f"  Usage entropy mean : {language_metrics['mean_usage_entropy']:.3f} / max {language_metrics['max_possible_entropy']:.3f}")
        print(f"  Entropy ratio : {language_metrics['entropy_ratio']:.2%}")
        print(f"  Concentration par lignée moy : {language_metrics['mean_token_lineage_concentration']:.2%}")
        print(f"  Distance L2 inter-lignées : {language_metrics['mean_inter_lineage_distance']:.2f}")
        print(f"  Per-token usage : {language_metrics['per_token_usage_top']}")
        print(f"\n  NOTE : pas de traduction sémantique, juste corrélations.")
        print(f"  'Un token émerge' = usage non aléatoire, concentré par lignée,")
        print(f"  corrélé à un contexte, et modifiant indirectement la survie.")

    # V8-B2.2 — Causalité comportementale
    if causality_tracker is not None:
        c = final_report.get("language_causality_v8b2_2", {})
        print(f"\n--- CAUSALITE COMPORTEMENTALE V8-B2.2 (test du verrou) ---")
        print(f"  Émissions totales : {c.get('n_emissions_total', 0)}")
        print(f"  Auditeurs totaux : {c.get('n_listeners_total', 0)}")
        print(f"  Listener shift KL moyen : {c.get('listener_shift_mean', 0):.4f}")
        print(f"  Listener shift KL max : {c.get('listener_shift_max', 0):.4f}")
        print(f"  Context consistency moyenne : {c.get('context_consistency_mean', 0):.2%}")
        print(f"  Per-token shift : {c.get('listener_shift_per_token', {})}")
        print(f"  Per-token context : {c.get('context_consistency_per_token', {})}")
        print(f"  VERDICT : {c.get('verdict', 'unknown')}")
        print(f"\n  Seuils : shift > 0.10 ET consistance > 0.50 = communication causale renforcée")
        print(f"           shift < 0.02 ET consistance < 0.30 = hypothèse décorative")

    # V8-C3 — Actions coopératives (gather_collective)
    coop_report = final_report.get("cooperative_v8c3", {})
    if coop_report.get("enabled"):
        print(f"\n--- COOPERATION V8-C3 (gather_collective) ---")
        print(f"  Gather SUCCESSES (≥2 agents adj + spot) : {coop_report.get('gather_successes_total', 0)}")
        print(f"  Gather FAILURES (tentatives ratées)      : {coop_report.get('gather_failures_total', 0)}")
        print(f"  Success rate : {coop_report.get('gather_success_rate', 0):.4f}")
        print(f"  Spots actifs (final) : {coop_report.get('active_spots_final', 0)}")
        print(f"  Bonus/spot : +{coop_report.get('bonus_energy', 0):.0f} energy")
        # Critère d'entrée curriculum C3a→C3b
        n_success = coop_report.get("gather_successes_total", 0)
        n_lin = len(final_report.get("final_state", {}).get("top_lineages", []))
        alive_final = final_report.get("final_state", {}).get("n_alive", 0)
        print(f"\n  --- Critère d'entrée C3a→C3b (ablation valide si tous OK) ---")
        print(f"  [{'OK' if n_success >= 50 else 'KO'}] gather_successes_total >= 50  (got {n_success})")
        print(f"  [{'OK' if n_lin >= 3 else 'KO'}] >= 3 lignées vivantes              (got {n_lin})")
        print(f"  [{'OK' if alive_final > 0 else 'KO'}] pas d'extinction               (alive={alive_final})")

        # V8-C3 — 4 métriques émergence coopération
        m = final_report.get("cooperative_metrics_v8c3", {})
        if m:
            print(f"\n  --- 4 métriques d'émergence (proto-coordination) ---")
            cl = m.get("clustering_pre_success", {})
            print(f"  [1] CLUSTERING avant succès :")
            print(f"      n_neighbors_r3 mean={cl.get('mean_neighbors_r3', 0):.2f} "
                  f"median={cl.get('median_neighbors_r3', 0)} "
                  f"trend(Q4-Q1)={cl.get('trend_q4_minus_q1', 0):+.2f}")
            dl = m.get("vocalize_to_gather_delay", {})
            print(f"  [2] VOCALIZE→GATHER delay :")
            print(f"      coverage={dl.get('coverage', 0):.2%} "
                  f"mean_min_delay={dl.get('mean_min_delay')} "
                  f"trend(Q4-Q1)={dl.get('trend_q4_minus_q1', 0):+.2f}  "
                  f"(<0 = apprentissage)")
            tk = m.get("token_entropy_pre_success", {})
            print(f"  [3] TOKEN ENTROPY pre-success :")
            print(f"      dominant_token={tk.get('dominant_token')} "
                  f"share={tk.get('dominant_share', 0):.2%} "
                  f"entropy={tk.get('entropy', 0):.3f} "
                  f"dist={tk.get('distribution', {})}")
            ch = m.get("success_chains", {})
            print(f"  [4] SUCCESS CHAINS (cascades) :")
            print(f"      n_chains={ch.get('n_chains', 0)} "
                  f"max_len={ch.get('max_chain_len', 0)} "
                  f"isolated={ch.get('n_isolated_successes', 0)} "
                  f"cascade_3+={ch.get('n_cascade_successes', 0)}")

    return final_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="results/v8b1_overnight")
    p.add_argument("--snap-every", type=int, default=5000)
    p.add_argument(
        "--regime", default="training",
        choices=[
            "training", "darwinian", "niches", "speciation",
            "language", "coordination", "coordination_hidden",
            "coordination_hard", "coordination_collective",
        ],
    )
    p.add_argument(
        "--vocalize-disable-after", type=int, default=None,
        help="V8-B2.3 — Test d'ablation : si défini, vocalize devient "
             "no-op après ce tick. Pour tester la fonction du langage.",
    )
    p.add_argument(
        "--vocalize-cost", type=float, default=0.05,
        help="V8-C3 M — Coût énergétique de vocalize. Défaut 0.05. "
             "Mettre à 0.001 pour tester ablation sans biais énergétique.",
    )
    p.add_argument(
        "--max-pop-override", type=int, default=None,
        help="V8-C3 P1 — Override max_population. Ex: 50 pour curriculum "
             "P1 (densité encore plus basse).",
    )
    p.add_argument(
        "--bonus-energy-override", type=float, default=None,
        help="V8-C3 P1 — Override CooperativeConfig.bonus_energy. "
             "Ex: 150.0 pour P1 (récompense renforcée).",
    )
    p.add_argument(
        "--n-initial-affinities", type=int, default=4,
        help="V8-C3 C2 — Nb d'affinités assignées aux fondateurs (1=mono, "
             "4=multi/défaut). Test causal diversité d'affinité.",
    )
    p.add_argument(
        "--n-seed-points", type=int, default=8,
        help="V8-C3 topology — Nb de seeds Voronoi (granularite spatiale). "
             "4=grosses regions, 8=defaut, 16=patchwork.",
    )
    args = p.parse_args()
    run_overnight(
        n_ticks=args.ticks, seed=args.seed, device=args.device,
        out_dir=args.out_dir, snap_every=args.snap_every,
        regime=args.regime,
        disable_vocalize_after_tick=args.vocalize_disable_after,
        vocalize_energy_cost=args.vocalize_cost,
        max_pop_override=args.max_pop_override,
        bonus_energy_override=args.bonus_energy_override,
        n_initial_affinities=args.n_initial_affinities,
        n_seed_points=args.n_seed_points,
    )


if __name__ == "__main__":
    main()
