"""GUI live pygame pour AetherLife V3 — multi-agent + saisons + heatmap température.

Contrôles :
    SPACE   pause / reprise
    R       reset épisode
    A       cycle policy (Random)
    N       cycle densité agents (2/4/8/16/32/64)
    T       toggle heatmap température
    ↑/↓     accélérer / ralentir
    ESC/Q   quitter
"""
from __future__ import annotations

import sys
from collections import deque
from typing import Callable

import numpy as np
import pygame

from aetherlife.metrics.episode_report import (
    EpisodeStatsTracker,
    format_report_lines,
)
from aetherlife.viz.report_overlay import render_report_overlay
from aetherlife.world.seasonal_grid import (
    Season,
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


# ─── viz config (V3.8 — living sandbox) ────────────────────────────────────
TRAIL_LEN = 14                  # nombre de positions précédentes affichées par agent
EAT_FLASH_FRAMES = 10           # durée du flash quand un agent mange
AGE_GRAY = (190, 190, 195)      # couleur de "vieillissement" (mix progressif)
MAX_AGE_MIX = 0.45              # max 45 % gris quand l'agent atteint max_steps
HALO_EXTRA = 5                  # rayon halo extra (proportionnel à l'énergie)


# ─── palette ───────────────────────────────────────────────────────────────
BG = (18, 18, 22)
GRID_LINE = (40, 40, 50)
CELL_FREE = (28, 30, 36)
CELL_FOOD = (90, 200, 90)
AGENT_DEAD = (90, 50, 50)
HUD_FG = (220, 220, 230)
HUD_DIM = (130, 130, 140)
HUD_RED = (220, 90, 90)
HUD_GREEN = (90, 200, 90)
HUD_PANEL = (24, 24, 30)

AGENT_COLORS = [
    (240, 180, 60), (60, 180, 240), (240, 80, 140), (180, 240, 60),
    (200, 100, 220), (240, 130, 60), (60, 220, 180), (200, 200, 80),
    (140, 100, 240), (240, 80, 80), (80, 240, 100), (220, 200, 220),
    (140, 200, 240), (220, 140, 80), (160, 240, 200), (240, 160, 200),
]

SEASON_LABELS = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}
SEASON_TINTS = {
    Season.SPRING: (140, 220, 140),
    Season.SUMMER: (240, 200, 100),
    Season.AUTUMN: (220, 140, 80),
    Season.WINTER: (140, 180, 240),
}


def _temp_to_color(temp: float, tmin: float, tmax: float) -> tuple[int, int, int]:
    """Map température dans [-10°C, 30°C] vers RGB bleu→rouge."""
    span = tmax - tmin
    norm = max(0.0, min(1.0, (temp - tmin) / span if span > 0 else 0.5))
    # bleu glacé (40, 80, 200) → rouge chaud (220, 80, 60)
    r = int(40 + (220 - 40) * norm)
    g = int(80 + (80 - 80) * norm)
    b = int(200 + (60 - 200) * norm)
    return (r, g, b)


def _random_policy_factory(env: SeasonalMultiAgentFoodGrid, seed: int = 0):
    rng = np.random.default_rng(seed)

    class _R:
        def act_dict(self, obs_dict: dict, *, greedy: bool = False) -> dict[int, int]:
            return {aid: int(rng.integers(0, 4)) for aid in obs_dict}

    return _R()


def run_gui_v3(
    base_cfg: SeasonalMultiAgentConfig | None = None,
    *,
    n_choices: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    cell_px: int = 18,
    hud_h: int = 140,
    tick_delay_ms: int = 60,
    seed: int = 0,
    show_temp_heatmap: bool = True,
) -> None:
    base_cfg = base_cfg or SeasonalMultiAgentConfig()

    def make_env(n_agents: int) -> SeasonalMultiAgentFoodGrid:
        cfg = SeasonalMultiAgentConfig(
            rows=base_cfg.rows, cols=base_cfg.cols, n_agents=n_agents,
            max_energy=base_cfg.max_energy, start_energy=base_cfg.start_energy,
            metabolism=base_cfg.metabolism, food_value=base_cfg.food_value,
            death_penalty=base_cfg.death_penalty,
            initial_food_density=base_cfg.initial_food_density,
            food_respawn_lambda=base_cfg.food_respawn_lambda,
            max_steps=base_cfg.max_steps,
            seasonal=base_cfg.seasonal,
        )
        return SeasonalMultiAgentFoodGrid(cfg)

    n_idx = n_choices.index(base_cfg.n_agents) if base_cfg.n_agents in n_choices else 2
    env = make_env(n_choices[n_idx])
    obs_dict, _ = env.reset(seed=seed)

    pygame.init()
    pygame.display.set_caption("AetherLife V3 — Seasonal Multi-Agent")
    width = base_cfg.cols * cell_px
    height = base_cfg.rows * cell_px + hud_h
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font_lg = pygame.font.SysFont("consolas", 18, bold=True)
    font_sm = pygame.font.SysFont("consolas", 13)

    policy = _random_policy_factory(env, seed=seed)
    show_temp = show_temp_heatmap
    paused = False
    delay_ms = tick_delay_ms
    episode_idx = 0
    last_outcome = "—"
    showing_report = False
    report_lines: list[str] = []
    tracker = EpisodeStatsTracker(n_agents=env.cfg.n_agents, track_seasons=True)
    tracker.reset(env)
    # --- V3.8 living sandbox state ---
    trails: dict[int, deque] = {
        a.agent_id: deque(maxlen=TRAIL_LEN) for a in env._agents  # noqa: SLF001
    }
    eat_flash_frames: dict[int, int] = {}
    show_trails = True

    def reset_episode(new_seed: int | None = None) -> None:
        nonlocal obs_dict, tracker, showing_report, trails, eat_flash_frames
        s = new_seed if new_seed is not None else seed + episode_idx
        obs_dict, _ = env.reset(seed=s)
        tracker = EpisodeStatsTracker(n_agents=env.cfg.n_agents, track_seasons=True)
        tracker.reset(env)
        showing_report = False
        trails = {a.agent_id: deque(maxlen=TRAIL_LEN) for a in env._agents}  # noqa: SLF001
        eat_flash_frames = {}

    def switch_density() -> None:
        nonlocal n_idx, env, policy, obs_dict, episode_idx, last_outcome
        n_idx = (n_idx + 1) % len(n_choices)
        env = make_env(n_choices[n_idx])
        policy = _random_policy_factory(env, seed=seed)
        episode_idx += 1
        reset_episode()
        last_outcome = "—"

    def end_episode(outcome: str) -> None:
        nonlocal last_outcome, showing_report, report_lines, episode_idx
        last_outcome = outcome
        report = tracker.finalize(env)
        report_lines = format_report_lines(report)
        showing_report = True
        episode_idx += 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    if showing_report:
                        reset_episode()
                    else:
                        paused = not paused
                elif event.key == pygame.K_r:
                    if not showing_report:
                        episode_idx += 1
                        last_outcome = "—"
                    reset_episode()
                elif event.key == pygame.K_n:
                    switch_density()
                elif event.key == pygame.K_t:
                    show_temp = not show_temp
                elif event.key == pygame.K_l:
                    show_trails = not show_trails
                elif event.key == pygame.K_UP:
                    delay_ms = max(0, delay_ms - 15)
                elif event.key == pygame.K_DOWN:
                    delay_ms = min(500, delay_ms + 15)

        if not paused and not showing_report and env.n_alive > 0 and env.step_count < env.cfg.max_steps:
            actions = policy.act_dict(obs_dict, greedy=True)
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            tracker.on_step(env, infos)
            # V3.8 — trail update + eat flashes
            for a in env._agents:  # noqa: SLF001
                if a.alive:
                    trails.setdefault(a.agent_id, deque(maxlen=TRAIL_LEN)).append(a.pos)
            for aid, info in infos.items():
                if info.get("ate"):
                    eat_flash_frames[aid] = EAT_FLASH_FRAMES
            obs_dict = {aid: o for aid, o in next_obs.items() if env.agent_state(aid).alive}
            if env.n_alive == 0 or env.step_count >= env.cfg.max_steps:
                outcome = (
                    "ALL DEAD" if env.n_alive == 0
                    else f"TRUNCATED ({env.n_alive}/{env.cfg.n_agents} alive)"
                )
                end_episode(outcome)

        # ─── render grid ───────────────────────────────────────────────────
        screen.fill(BG)
        temp_field = env.temperature_field
        tmin = env.cfg.seasonal.temp_min
        tmax = env.cfg.seasonal.temp_max

        for r in range(env.cfg.rows):
            for c in range(env.cfg.cols):
                x, y = c * cell_px, r * cell_px
                rect = pygame.Rect(x, y, cell_px, cell_px)
                if env.food_mask[r, c]:
                    color = CELL_FOOD
                elif show_temp:
                    color = _temp_to_color(float(temp_field[r, c]), tmin, tmax)
                else:
                    color = CELL_FREE
                pygame.draw.rect(screen, color, rect)
                if cell_px >= 12:
                    pygame.draw.rect(screen, GRID_LINE, rect, 1)

        # V3.8 — draw trails first (under agents)
        if show_trails:
            for aid, trail in trails.items():
                base_color = AGENT_COLORS[aid % len(AGENT_COLORS)]
                trail_list = list(trail)
                trail_n = len(trail_list)
                if trail_n <= 1:
                    continue
                for i, (tr, tc) in enumerate(trail_list[:-1]):
                    # Plus i grand = plus récent = plus opaque
                    alpha = int(180 * (i + 1) / trail_n)
                    tcx = tc * cell_px + cell_px // 2
                    tcy = tr * cell_px + cell_px // 2
                    tr_radius = max(1, cell_px // 7)
                    surf = pygame.Surface((tr_radius * 2, tr_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(
                        surf, (*base_color, alpha),
                        (tr_radius, tr_radius), tr_radius,
                    )
                    screen.blit(surf, (tcx - tr_radius, tcy - tr_radius))

        # V3.8 — draw agents with energy halo + age tint + eat flash
        max_steps_safe = max(env.cfg.max_steps, 1)
        age_frac = min(1.0, env.step_count / max_steps_safe)
        for a in env._agents:  # noqa: SLF001
            ar, ac = a.pos
            cx = ac * cell_px + cell_px // 2
            cy = ar * cell_px + cell_px // 2
            radius = max(2, cell_px // 2 - 2)
            if not a.alive:
                pygame.draw.circle(screen, AGENT_DEAD, (cx, cy), radius)
                pygame.draw.circle(screen, (50, 30, 30), (cx, cy), radius, 1)
                continue
            base_color = AGENT_COLORS[a.agent_id % len(AGENT_COLORS)]
            # Age tint : mix vers gris à mesure que l'agent vieillit
            mix = MAX_AGE_MIX * age_frac
            color = tuple(
                int(base_color[i] * (1 - mix) + AGE_GRAY[i] * mix) for i in range(3)
            )
            # Energy halo (couronne externe, alpha proportionnel à energy)
            e_frac = max(0.0, min(1.0, a.energy / env.cfg.max_energy))
            halo_radius = radius + int(HALO_EXTRA * e_frac)
            halo_alpha = int(40 + 80 * e_frac)
            halo_surf = pygame.Surface(
                (halo_radius * 2, halo_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                halo_surf, (*color, halo_alpha),
                (halo_radius, halo_radius), halo_radius,
            )
            screen.blit(halo_surf, (cx - halo_radius, cy - halo_radius))
            # Eat flash (anneau pulsant)
            if eat_flash_frames.get(a.agent_id, 0) > 0:
                progress = 1.0 - (eat_flash_frames[a.agent_id] / EAT_FLASH_FRAMES)
                flash_r = radius + int(2 + 12 * progress)
                flash_alpha = int(220 * (1 - progress))
                flash_surf = pygame.Surface(
                    (flash_r * 2 + 2, flash_r * 2 + 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    flash_surf, (255, 255, 200, flash_alpha),
                    (flash_r + 1, flash_r + 1), flash_r, 2,
                )
                screen.blit(flash_surf, (cx - flash_r - 1, cy - flash_r - 1))
                eat_flash_frames[a.agent_id] -= 1
            # Body
            pygame.draw.circle(screen, color, (cx, cy), radius)
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), radius, 1)
            # Inner energy dot
            inner_r = max(1, int(radius * e_frac * 0.7))
            if inner_r >= 2:
                inner_color = (
                    min(255, color[0] + 30),
                    min(255, color[1] + 30),
                    min(255, color[2] + 30),
                )
                pygame.draw.circle(screen, inner_color, (cx, cy), inner_r)

        # ─── HUD ───────────────────────────────────────────────────────────
        hud_y0 = env.cfg.rows * cell_px
        pygame.draw.rect(screen, HUD_PANEL, pygame.Rect(0, hud_y0, width, hud_h))

        season_name = SEASON_LABELS[int(env.season)]
        season_tint = SEASON_TINTS[env.season]
        alive_rate = env.n_alive / env.cfg.n_agents
        outcome_color = HUD_RED if last_outcome == "ALL DEAD" else (
            HUD_GREEN if last_outcome.startswith("TRUNCATED") else HUD_DIM
        )
        period = env.cfg.seasonal.season_period
        step_in_season = env.step_count % period
        ticks_to_next = period // 4 - (step_in_season % (period // 4))

        line1 = (
            f"season={season_name:6s}  phase={env.phase:5.2f}  "
            f"next-in={ticks_to_next:3d}t  step={env.step_count:4d}/{env.cfg.max_steps}"
        )
        line2 = (
            f"N={env.cfg.n_agents:3d}  alive={env.n_alive:3d} ({alive_rate:5.1%})  "
            f"food={env.food_count:4d}  episode#{episode_idx}"
        )
        line3 = f"last: {last_outcome}"
        line4 = "Mean food regen factors per season:"
        line5 = (
            f"  spring={env.cfg.seasonal.spring_lambda_factor:.2f}  "
            f"summer={env.cfg.seasonal.summer_lambda_factor:.2f}  "
            f"autumn={env.cfg.seasonal.autumn_lambda_factor:.2f}  "
            f"winter={env.cfg.seasonal.winter_lambda_factor:.2f}"
        )
        controls = (
            "SPACE pause/next  R reset  N density  T heatmap  L trails  "
            "↑/↓ speed  Q quit"
        )

        screen.blit(font_lg.render(line1, True, season_tint), (12, hud_y0 + 8))
        screen.blit(font_sm.render(line2, True, HUD_FG), (12, hud_y0 + 32))
        screen.blit(font_sm.render(line3, True, outcome_color), (12, hud_y0 + 50))
        screen.blit(font_sm.render(line5, True, HUD_DIM), (12, hud_y0 + 80))
        screen.blit(font_sm.render(controls, True, HUD_DIM), (12, hud_y0 + 110))

        if paused:
            screen.blit(font_lg.render("PAUSED", True, HUD_RED), (width - 90, hud_y0 + 8))

        if showing_report:
            render_report_overlay(screen, report_lines, font_lg, font_sm)

        pygame.display.flip()
        clock.tick(60)
        if delay_ms > 0 and not paused and not showing_report:
            pygame.time.wait(delay_ms)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_gui_v3()
