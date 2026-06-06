"""OBS Viewer 2.0 (lite) — observateur LIVE de la simulation V8.

Réutilise le rendu V8 (`pygame_viewer_v8._draw_frame`) et la sim
(`build_env`+`LineageAgent`). Touches : ESPACE pause, +/- jours, H Historien,
E export, ↑/↓ vitesse, ESC quitter. 1 jour = ticks_per_day ticks.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys

import numpy as np
import pygame

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "scripts"),
)

from overnight_v8b1 import build_env  # noqa: E402
from aetherlife.agents.lineage_agent import LineageAgent  # noqa: E402
from aetherlife.agents.lineage_brain import BrainConfig  # noqa: E402
from aetherlife.historian import Historian  # noqa: E402
from aetherlife.historian.spatial_mobility import (  # noqa: E402
    OccupancyAccumulator, build_spatial_mobility_block, window_bounds,
)
from aetherlife.viz.live_report import build_live_report  # noqa: E402
from aetherlife.viz.pygame_viewer_v8 import _draw_frame  # noqa: E402

_HUD2 = 22
_OVL_BG = (12, 12, 16)
_FG = (215, 215, 220)


def build_event_dict(env, policy, t: int) -> dict:
    """Event tick au schéma du recorder (consommé par _draw_frame)."""
    agents = [
        {"id": a.agent_id, "lin": a.root_ancestor_id,
         "r": a.pos[0], "c": a.pos[1], "e": round(float(a.energy), 1)}
        for a in env._agents if a.alive  # noqa: SLF001
    ]
    vocal = {str(s): int(tk)
             for s, tk in getattr(env, "_tokens_this_tick", {}).items()}
    spots = []
    for pos in getattr(env, "gather_spots", []):
        n = sum(1 for a in env._agents  # noqa: SLF001
                if a.alive and abs(a.pos[0] - pos[0]) + abs(a.pos[1] - pos[1]) <= 1)
        spots.append({"r": pos[0], "c": pos[1], "n": n})
    return {"t": t, "season": int(env.season), "n_alive": env.n_alive,
            "n_lin": len(policy.registry), "agents": agents,
            "vocal": vocal, "spots": spots}


def render_live_frame(event, meta, *, hud_extra: str, cell_px: int):
    """Frame V8 (_draw_frame) + une ligne de HUD live en bas."""
    frame = _draw_frame(event, meta, cell_px)
    w, h = frame.get_width(), frame.get_height()
    surf = pygame.Surface((w, h + _HUD2))
    surf.fill((0, 0, 0))
    surf.blit(frame, (0, 0))
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont("monospace", 13)
    surf.blit(font.render(hud_extra, True, _FG), (6, h + 4))
    return surf


def _draw_historian_overlay(screen, lines):
    w, h = screen.get_size()
    panel = pygame.Surface((w, h))
    panel.set_alpha(235)
    panel.fill(_OVL_BG)
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont("monospace", 14)
    for i, line in enumerate(lines[:40]):
        panel.blit(font.render(line[:110], True, _FG), (12, 10 + i * 18))
    screen.blit(panel, (0, 0))


def _historian_lines(env, policy, occ_s, occ_e, windows, n_ticks):
    rep = build_live_report(env, policy, occ_s, occ_e,
                            windows=windows, n_ticks=n_ticks)
    h = Historian.from_report(rep, run_id="live")
    sm = rep["spatial_mobility_v8c3"]
    lines = [
        "=== COMPTE RENDU HISTORIEN (live) ===  [H ferme · E exporte]",
        f"pop={rep['final_state']['n_alive']}  "
        f"births={rep['final_state']['n_births_total']}  "
        f"lignees={rep['criterion_3_selection']['n_lineages_final']}",
        f"vocalize={rep['language_metrics_v8b2'].get('total_vocalize_count', 0)}  "
        f"gather={rep['cooperative_v8c3']['gather_successes_total']}  "
        f"mobility={sm.get('corr_occupation_start_end')}",
        "",
        f"--- DECOUVERTES ({len(h.discoveries)}) ---",
    ]
    for d in h.discoveries:
        lines.append(f"[{d.confidence:.2f}] {d.slug}")
        lines.append(f"     {d.headline}")
    if not h.discoveries:
        lines.append("(aucun pattern significatif pour l'instant)")
    return lines, h


def run_live(*, seed: int = 1, regime: str = "coordination_collective",
             device: str = "cuda", days: int = 5, ticks_per_day: int = 1000,
             cell_px: int = 14, delay_ms: int = 30,
             max_frames: int | None = None) -> None:
    env = build_env(seed, regime=regime, vocalize_energy_cost=0.05)
    vision_radius = 2 if regime in (
        "coordination", "coordination_hidden", "coordination_hard") else 4
    cfg = BrainConfig(
        enabled=True, device=device, vision_radius=vision_radius,
        hidden_dims=(64, 64), lr=1e-4, batch_size=64, buffer_capacity=50_000,
        min_replay_to_learn=500, train_every=4, epsilon_start=0.6,
        epsilon_end=0.08, epsilon_decay_steps=30_000, target_sync_steps=200,
        mutation_std=0.03,
    )
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=seed)
    meta = {"rows": env.cfg.rows, "cols": env.cfg.cols,
            "n_tokens": env.cfg.vocabulary.n_tokens}

    pygame.init()
    pygame.font.init()
    probe = render_live_frame(build_event_dict(env, policy, 0), meta,
                              hud_extra="", cell_px=cell_px)
    screen = pygame.display.set_mode(probe.get_size())
    pygame.display.set_caption("AetherLife — Live V8 Observer")

    occ_s = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    occ_e = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    t = 0
    paused = False
    overlay = None  # None ou (lines, historian)
    frames = 0
    running = True
    while running:
        budget = days * ticks_per_day
        swin, ewin = window_bounds(budget)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS,
                                   pygame.K_EQUALS):
                    days += 1
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    days = max(1, days - 1)
                elif event.key == pygame.K_h:
                    if overlay is None:
                        overlay = _historian_lines(env, policy, occ_s, occ_e,
                                                   (swin, ewin), max(t, 1))
                    else:
                        overlay = None
                elif event.key == pygame.K_e:
                    lines, h = _historian_lines(env, policy, occ_s, occ_e,
                                                (swin, ewin), max(t, 1))
                    h.write_all("results/gui_run/report")
                    overlay = (lines + ["", ">>> exporte: results/gui_run/report"],
                               h)
                elif event.key == pygame.K_UP:
                    delay_ms = max(0, delay_ms - 15)
                elif event.key == pygame.K_DOWN:
                    delay_ms = min(500, delay_ms + 15)

        if not paused and overlay is None and env.n_alive > 0 and t < budget:
            t += 1
            obs = {a.agent_id: np.zeros(10)
                   for a in env._agents if a.alive}  # noqa: SLF001
            env.step(policy.act_dict(obs, greedy=False))
            if swin[0] < t <= swin[1] or ewin[0] < t <= ewin[1]:
                pos = [(a.pos[0], a.pos[1])
                       for a in env._agents if a.alive]  # noqa: SLF001
                (occ_s if t <= swin[1] else occ_e).add_positions(pos)

        ev = build_event_dict(env, policy, t)
        day_cur = t // ticks_per_day + (1 if t % ticks_per_day else 0)
        hud = (f"day {min(day_cur, days)}/{days}  pop={env.n_alive}  "
               f"births={env.n_births_total}  "
               f"gather={env.gather_successes_total}  "
               f"{'PAUSE' if paused else ''}{' FIN' if t >= budget else ''}  "
               f"[H]istorien [E]xport [+/-]jours")
        frame = render_live_frame(ev, meta, hud_extra=hud, cell_px=cell_px)
        screen.blit(frame, (0, 0))
        if overlay is not None:
            _draw_historian_overlay(screen, overlay[0])
        pygame.display.flip()
        if delay_ms and not paused and overlay is None:
            pygame.time.delay(delay_ms)

        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False
    pygame.display.quit()
