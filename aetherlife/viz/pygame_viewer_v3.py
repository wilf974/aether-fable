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
from typing import Callable

import numpy as np
import pygame

from aetherlife.world.seasonal_grid import (
    Season,
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


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

    def reset_episode(new_seed: int | None = None) -> None:
        nonlocal obs_dict
        s = new_seed if new_seed is not None else seed + episode_idx
        obs_dict, _ = env.reset(seed=s)

    def switch_density() -> None:
        nonlocal n_idx, env, policy, obs_dict, episode_idx, last_outcome
        n_idx = (n_idx + 1) % len(n_choices)
        env = make_env(n_choices[n_idx])
        policy = _random_policy_factory(env, seed=seed)
        episode_idx += 1
        reset_episode()
        last_outcome = "—"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    episode_idx += 1
                    reset_episode()
                    last_outcome = "—"
                elif event.key == pygame.K_n:
                    switch_density()
                elif event.key == pygame.K_t:
                    show_temp = not show_temp
                elif event.key == pygame.K_UP:
                    delay_ms = max(0, delay_ms - 15)
                elif event.key == pygame.K_DOWN:
                    delay_ms = min(500, delay_ms + 15)

        if not paused and env.n_alive > 0 and env.step_count < env.cfg.max_steps:
            actions = policy.act_dict(obs_dict, greedy=True)
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            obs_dict = {aid: o for aid, o in next_obs.items() if env.agent_state(aid).alive}
            if env.n_alive == 0 or env.step_count >= env.cfg.max_steps:
                last_outcome = (
                    "ALL DEAD" if env.n_alive == 0
                    else f"TRUNCATED ({env.n_alive}/{env.cfg.n_agents} alive)"
                )
                episode_idx += 1
                pygame.time.wait(500)
                reset_episode()

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

        for a in env._agents:  # noqa: SLF001
            ar, ac = a.pos
            cx = ac * cell_px + cell_px // 2
            cy = ar * cell_px + cell_px // 2
            radius = max(2, cell_px // 2 - 2)
            color = AGENT_DEAD if not a.alive else AGENT_COLORS[a.agent_id % len(AGENT_COLORS)]
            pygame.draw.circle(screen, color, (cx, cy), radius)
            if a.alive:
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy), radius, 1)
                e_frac = max(0.0, a.energy / env.cfg.max_energy)
                inner_r = max(1, int(radius * e_frac))
                if inner_r > 1:
                    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), inner_r, 1)

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
        controls = "SPACE pause  R reset  N density  T toggle heatmap  ↑/↓ speed  Q quit"

        screen.blit(font_lg.render(line1, True, season_tint), (12, hud_y0 + 8))
        screen.blit(font_sm.render(line2, True, HUD_FG), (12, hud_y0 + 32))
        screen.blit(font_sm.render(line3, True, outcome_color), (12, hud_y0 + 50))
        screen.blit(font_sm.render(line5, True, HUD_DIM), (12, hud_y0 + 80))
        screen.blit(font_sm.render(controls, True, HUD_DIM), (12, hud_y0 + 110))

        if paused:
            screen.blit(font_lg.render("PAUSED", True, HUD_RED), (width - 90, hud_y0 + 8))

        pygame.display.flip()
        clock.tick(60)
        if delay_ms > 0 and not paused:
            pygame.time.wait(delay_ms)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_gui_v3()
