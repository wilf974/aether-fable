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
from aetherlife.world.biomes import BiomeConfig
from aetherlife.world.cache import CacheConfig
from aetherlife.world.competition import CompetitionConfig
from aetherlife.world.construction import BuildConfig
from aetherlife.world.planting import PlantingConfig
from aetherlife.world.reproduction import ReproductionConfig
from aetherlife.world.seasonal_grid import (
    SeasonalConfig, SeasonalMultiAgentConfig, SeasonalMultiAgentFoodGrid,
)
from aetherlife.world.vocabulary import VocabularyConfig


def build_env(seed: int, *, regime: str = "training") -> SeasonalMultiAgentFoodGrid:
    """Config selon regime :

    - 'training' : permissif, monde homogène (V8-B1).
    - 'darwinian' : dur, monde homogène (V8-B1).
    - 'niches'   : V8-B1.5 — biomes Voronoi + compétition locale +
                   max_pop élevé. Cible : ≥3 lignées coexistantes après 100k.
    """
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
    else:
        rows = 30
        cols = 30
        n_agents = 10
        max_pop = 30
        food_respawn_lambda = 0.25
        biome_cfg = BiomeConfig(enabled=False)
        competition_cfg = CompetitionConfig(enabled=False)
    # V8-B2.0 — vocabulary activée seulement en mode language
    if regime == "language":
        vocab_cfg = VocabularyConfig(
            enabled=True, n_tokens=4, embedding_dim=16,
            listen_radius=5, mutation_std=0.05,
            vocalize_energy_cost=0.05, social_bonus=0.0,
        )
    else:
        vocab_cfg = VocabularyConfig(enabled=False)
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
) -> dict:
    env = build_env(seed, regime=regime)
    print(f"REGIME={regime}")
    # V8-B2.1 — re-stabilisation pour action space étendu (8 actions
    # avec langage). lr plus bas, target sync plus fréquent, epsilon_end
    # légèrement >0 pour exploration résiduelle.
    cfg = BrainConfig(
        enabled=True, device=device, vision_radius=4,
        hidden_dims=(64, 64), lr=1e-4, batch_size=64,
        buffer_capacity=50_000, min_replay_to_learn=500, train_every=4,
        epsilon_start=0.6, epsilon_end=0.08, epsilon_decay_steps=30_000,
        target_sync_steps=200, mutation_std=0.03,
    )
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=seed)
    print(f"OBS_DIM={policy.obs_dim}  device={cfg.device}  brains={len(policy.registry)}")

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

    for t in range(1, n_ticks + 1):
        if env.n_alive == 0:
            print(f"[t={t}] EXTINCTION")
            break

        obs_stub = {a.agent_id: np.zeros(10) for a in env._agents if a.alive}  # noqa: SLF001
        actions = policy.act_dict(obs_stub, greedy=False)
        e_before = {
            a.agent_id: a.energy for a in env._agents if a.alive  # noqa: SLF001
        }
        env.step(actions)

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
            lineages_curve.append((t, len(policy.registry)))
            if not np.isnan(mean_loss):
                loss_curve.append((t, mean_loss))

            # Cull periodic
            alive_roots = {a.root_ancestor_id for a in env._agents if a.alive}  # noqa: SLF001
            policy.registry.cull_dead_lineages(alive_roots)

        if t % divergence_every == 0:
            div = measure_policy_divergence(policy, n_samples=64)
            divergence_curve.append((t, div))

    dt = time.time() - t0

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
        "curves": {
            "alive": alive_curve,
            "lineages": lineages_curve,
            "divergence": divergence_curve,
            "loss": loss_curve,
        },
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
        choices=["training", "darwinian", "niches", "speciation", "language"],
    )
    args = p.parse_args()
    run_overnight(
        n_ticks=args.ticks, seed=args.seed, device=args.device,
        out_dir=args.out_dir, snap_every=args.snap_every,
        regime=args.regime,
    )


if __name__ == "__main__":
    main()
