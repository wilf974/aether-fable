"""GUI live pygame pour AetherLife V1/V1.5 — visualisation Solo Forager.

Contrôles :
    SPACE   pause / reprise
    R       reset épisode
    A       switch agent (Greedy / Random / DQN trained si checkpoint fourni)
    ↑/↓     vitesse +/-  (delay ticks)
    ESC/Q   quitter
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pygame

from aetherlife.agents.greedy_agent import GreedyAgent
from aetherlife.agents.random_agent import RandomAgent
from aetherlife.config import FoodGridConfig
from aetherlife.metrics.episode_report import (
    EpisodeStatsTracker,
    format_report_lines,
)
from aetherlife.viz.report_overlay import render_report_overlay
from aetherlife.world.food_grid import FoodGrid


# ─── palette ───────────────────────────────────────────────────────────────
BG = (18, 18, 22)
GRID_LINE = (40, 40, 50)
CELL_FREE = (28, 30, 36)
CELL_FOOD = (90, 200, 90)
AGENT_BODY = (240, 180, 60)
AGENT_OUTLINE = (255, 220, 120)
HUD_FG = (220, 220, 230)
HUD_DIM = (130, 130, 140)
HUD_RED = (220, 90, 90)
HUD_GREEN = (90, 200, 90)
ENERGY_BAR_BG = (50, 50, 60)
ENERGY_BAR_FG = (240, 180, 60)


class _ActAgent(Protocol):
    def act(self, observation: np.ndarray, *, info: dict | None = None) -> int: ...


@dataclass
class _AgentSlot:
    name: str
    factory: callable  # type: ignore[type-arg]


def _build_agents(
    cfg: FoodGridConfig,
    seed: int,
    env: FoodGrid,
    dqn_checkpoint: Path | None = None,
) -> list[_AgentSlot]:
    slots = [
        _AgentSlot("Greedy", lambda: GreedyAgent(rows=cfg.rows, cols=cfg.cols, seed=seed)),
        _AgentSlot("Random", lambda: RandomAgent(n_actions=4, seed=seed)),
    ]
    if dqn_checkpoint is not None and dqn_checkpoint.exists():
        from mw_ia.config import DQNConfig

        from aetherlife.agents.dqn_agent import DQNAgent

        def make_dqn() -> "_GreedyDQNWrapper":
            dqn = DQNAgent(env, DQNConfig(), device="cpu", seed=seed)
            dqn.load(dqn_checkpoint)
            return _GreedyDQNWrapper(dqn)

        slots.append(_AgentSlot("DQN", make_dqn))
    return slots


class _GreedyDQNWrapper:
    """Wrap un DQNAgent pour qu'il joue toujours greedy dans la GUI."""

    def __init__(self, dqn: "object") -> None:
        self._dqn = dqn

    def act(self, observation: np.ndarray, *, info: dict | None = None) -> int:
        return self._dqn.act(observation, greedy=True)  # type: ignore[attr-defined]


def run_gui(
    cfg: FoodGridConfig | None = None,
    *,
    cell_px: int = 32,
    hud_h: int = 100,
    tick_delay_ms: int = 80,
    seed: int = 0,
    dqn_checkpoint: Path | None = None,
) -> None:
    cfg = cfg or FoodGridConfig()
    env = FoodGrid(cfg)
    obs, _ = env.reset(seed=seed)

    pygame.init()
    pygame.display.set_caption("AetherLife V1 — Solo Forager")
    width = cfg.cols * cell_px
    height = cfg.rows * cell_px + hud_h
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font_lg = pygame.font.SysFont("consolas", 18, bold=True)
    font_sm = pygame.font.SysFont("consolas", 13)

    agents = _build_agents(cfg, seed=seed, env=env, dqn_checkpoint=dqn_checkpoint)
    slot_idx = 0
    agent: _ActAgent = agents[slot_idx].factory()

    paused = False
    delay_ms = tick_delay_ms
    episode_idx = 0
    last_outcome = "—"
    last_lifespan = 0
    last_food = 0
    total_reward = 0.0
    food_eaten = 0
    showing_report = False
    report_lines: list[str] = []
    tracker = EpisodeStatsTracker(n_agents=1)
    tracker.reset(env)

    def reset_episode(new_seed: int | None = None) -> None:
        nonlocal obs, total_reward, food_eaten, tracker, showing_report
        s = new_seed if new_seed is not None else seed + episode_idx
        obs, _ = env.reset(seed=s)
        total_reward = 0.0
        food_eaten = 0
        tracker = EpisodeStatsTracker(n_agents=1)
        tracker.reset(env)
        showing_report = False

    def end_episode(outcome: str) -> None:
        nonlocal last_outcome, last_lifespan, last_food, showing_report, report_lines, episode_idx
        last_outcome = outcome
        last_lifespan = env.step_count
        last_food = food_eaten
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
                elif event.key == pygame.K_a:
                    slot_idx = (slot_idx + 1) % len(agents)
                    agent = agents[slot_idx].factory()
                    episode_idx += 1
                    reset_episode()
                    last_outcome = "—"
                elif event.key == pygame.K_UP:
                    delay_ms = max(0, delay_ms - 20)
                elif event.key == pygame.K_DOWN:
                    delay_ms = min(500, delay_ms + 20)

        if not paused and not showing_report:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.on_step(env, {0: info})
            total_reward += reward
            if info.get("ate"):
                food_eaten += 1
            if terminated or truncated:
                end_episode("SURVIVED" if truncated else "DIED")

        # ─── render grid ───────────────────────────────────────────────────
        screen.fill(BG)
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                x, y = c * cell_px, r * cell_px
                rect = pygame.Rect(x, y, cell_px, cell_px)
                color = CELL_FOOD if env.food_mask[r, c] else CELL_FREE
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, 1)

        ar, ac = env.pos
        cx = ac * cell_px + cell_px // 2
        cy = ar * cell_px + cell_px // 2
        radius = cell_px // 2 - 4
        pygame.draw.circle(screen, AGENT_BODY, (cx, cy), radius)
        pygame.draw.circle(screen, AGENT_OUTLINE, (cx, cy), radius, 2)

        # ─── HUD ───────────────────────────────────────────────────────────
        hud_y0 = cfg.rows * cell_px
        pygame.draw.rect(screen, (24, 24, 30), pygame.Rect(0, hud_y0, width, hud_h))

        # bar énergie
        bar_x, bar_y = 12, hud_y0 + 14
        bar_w, bar_h = width - 24, 16
        pygame.draw.rect(screen, ENERGY_BAR_BG, pygame.Rect(bar_x, bar_y, bar_w, bar_h))
        e_frac = max(0.0, env.energy / cfg.max_energy)
        pygame.draw.rect(
            screen, ENERGY_BAR_FG,
            pygame.Rect(bar_x, bar_y, int(bar_w * e_frac), bar_h),
        )
        screen.blit(
            font_sm.render(
                f"energy {env.energy:6.1f}/{cfg.max_energy:.0f}",
                True, HUD_FG,
            ),
            (bar_x + 6, bar_y + 1),
        )

        # lignes infos
        info_y = bar_y + bar_h + 10
        agent_name = agents[slot_idx].name
        line1 = f"agent={agent_name:6s}  step={env.step_count:4d}/{cfg.max_steps}  food_in_grid={env.food_count:3d}"
        line2 = f"reward_cumul={total_reward:+8.1f}  food_eaten={food_eaten:3d}  episode#{episode_idx}"
        outcome_color = HUD_GREEN if last_outcome == "SURVIVED" else (HUD_RED if last_outcome == "DIED" else HUD_DIM)
        line3 = f"last episode: {last_outcome}  lifespan={last_lifespan}  food={last_food}"
        controls = "SPACE pause  R reset  A switch agent  ↑/↓ speed  Q/ESC quit"

        screen.blit(font_lg.render(line1, True, HUD_FG), (12, info_y))
        screen.blit(font_sm.render(line2, True, HUD_DIM), (12, info_y + 22))
        screen.blit(font_sm.render(line3, True, outcome_color), (12, info_y + 38))
        screen.blit(font_sm.render(controls, True, HUD_DIM), (12, info_y + 56))

        if paused:
            pause_text = font_lg.render("PAUSED", True, HUD_RED)
            screen.blit(pause_text, (width - 90, info_y))

        if showing_report:
            render_report_overlay(screen, report_lines, font_lg, font_sm)

        pygame.display.flip()
        clock.tick(60)
        if delay_ms > 0 and not paused and not showing_report:
            pygame.time.wait(delay_ms)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_gui()
