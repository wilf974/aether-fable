"""GUI live pygame pour AetherLife V2 — visualisation multi-agent.

Contrôles :
    SPACE   pause / reprise
    R       reset épisode
    A       cycle policy (Random / IDQN trained si checkpoint fourni)
    N       cycle densité agents (2 / 4 / 8 / 16 / 32 / 64)
    ↑/↓     accélérer / ralentir
    ESC/Q   quitter
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pygame

from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
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

# couleurs distinctes pour les agents (cycle si plus de 16)
AGENT_COLORS = [
    (240, 180, 60), (60, 180, 240), (240, 80, 140), (180, 240, 60),
    (200, 100, 220), (240, 130, 60), (60, 220, 180), (200, 200, 80),
    (140, 100, 240), (240, 80, 80), (80, 240, 100), (220, 200, 220),
    (140, 200, 240), (220, 140, 80), (160, 240, 200), (240, 160, 200),
]


@dataclass
class _PolicySlot:
    name: str
    factory: Callable[[MultiAgentFoodGrid], object]


def _random_policy_factory(env: MultiAgentFoodGrid, seed: int = 0):
    rng = np.random.default_rng(seed)

    class _R:
        def act_dict(self, obs_dict: dict, *, greedy: bool = False) -> dict[int, int]:
            return {aid: int(rng.integers(0, 4)) for aid in obs_dict}

    return _R()


def _build_policies(
    env: MultiAgentFoodGrid,
    seed: int,
    idqn_checkpoint: Path | None,
) -> list[_PolicySlot]:
    slots = [_PolicySlot("Random", lambda e: _random_policy_factory(e, seed=seed))]
    if idqn_checkpoint is not None and idqn_checkpoint.exists():
        from mw_ia.config import DQNConfig

        from aetherlife.agents.independent_dqn import IndependentDQNAgent

        def make_idqn(e: MultiAgentFoodGrid):
            ag = IndependentDQNAgent(e, DQNConfig(), device="cpu", seed=seed)
            ag.load(idqn_checkpoint)

            class _G:
                def act_dict(self, obs_dict: dict, *, greedy: bool = False) -> dict[int, int]:
                    return ag.act_dict(obs_dict, greedy=True)
            return _G()

        slots.append(_PolicySlot("IDQN", make_idqn))
    return slots


def run_gui_v2(
    base_cfg: MultiAgentForagerConfig | None = None,
    *,
    n_choices: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    cell_px: int = 18,
    hud_h: int = 130,
    tick_delay_ms: int = 60,
    seed: int = 0,
    idqn_checkpoint: Path | None = None,
) -> None:
    base_cfg = base_cfg or MultiAgentForagerConfig()

    def make_env(n_agents: int) -> MultiAgentFoodGrid:
        cfg = MultiAgentForagerConfig(
            rows=base_cfg.rows, cols=base_cfg.cols, n_agents=n_agents,
            max_energy=base_cfg.max_energy, start_energy=base_cfg.start_energy,
            metabolism=base_cfg.metabolism, food_value=base_cfg.food_value,
            death_penalty=base_cfg.death_penalty,
            initial_food_density=base_cfg.initial_food_density,
            food_respawn_lambda=base_cfg.food_respawn_lambda,
            max_steps=base_cfg.max_steps,
        )
        return MultiAgentFoodGrid(cfg)

    n_idx = n_choices.index(base_cfg.n_agents) if base_cfg.n_agents in n_choices else 2
    env = make_env(n_choices[n_idx])
    obs_dict, _ = env.reset(seed=seed)

    pygame.init()
    pygame.display.set_caption("AetherLife V2 — Multi-Agent Forager")
    width = base_cfg.cols * cell_px
    height = base_cfg.rows * cell_px + hud_h
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font_lg = pygame.font.SysFont("consolas", 18, bold=True)
    font_sm = pygame.font.SysFont("consolas", 13)

    policies = _build_policies(env, seed=seed, idqn_checkpoint=idqn_checkpoint)
    policy_idx = 0
    policy = policies[policy_idx].factory(env)

    paused = False
    delay_ms = tick_delay_ms
    episode_idx = 0
    last_outcome = "—"
    last_alive_final = 0

    def reset_episode(new_seed: int | None = None) -> None:
        nonlocal obs_dict
        s = new_seed if new_seed is not None else seed + episode_idx
        obs_dict, _ = env.reset(seed=s)

    def switch_density(delta: int) -> None:
        nonlocal n_idx, env, policy, obs_dict, episode_idx, last_outcome
        n_idx = (n_idx + delta) % len(n_choices)
        env = make_env(n_choices[n_idx])
        policy = policies[policy_idx].factory(env)
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
                elif event.key == pygame.K_a:
                    policy_idx = (policy_idx + 1) % len(policies)
                    policy = policies[policy_idx].factory(env)
                    episode_idx += 1
                    reset_episode()
                    last_outcome = "—"
                elif event.key == pygame.K_n:
                    switch_density(1)
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
                last_alive_final = env.n_alive
                episode_idx += 1
                # auto-restart après 30 frames de pause
                pygame.time.wait(500)
                reset_episode()

        # ─── render grid ───────────────────────────────────────────────────
        screen.fill(BG)
        for r in range(env.cfg.rows):
            for c in range(env.cfg.cols):
                x, y = c * cell_px, r * cell_px
                rect = pygame.Rect(x, y, cell_px, cell_px)
                color = CELL_FOOD if env.food_mask[r, c] else CELL_FREE
                pygame.draw.rect(screen, color, rect)
                if cell_px >= 12:
                    pygame.draw.rect(screen, GRID_LINE, rect, 1)

        # draw agents
        for a in env._agents:  # noqa: SLF001 — viz needs internal state
            ar, ac = a.pos
            cx = ac * cell_px + cell_px // 2
            cy = ar * cell_px + cell_px // 2
            radius = max(2, cell_px // 2 - 2)
            color = AGENT_DEAD if not a.alive else AGENT_COLORS[a.agent_id % len(AGENT_COLORS)]
            pygame.draw.circle(screen, color, (cx, cy), radius)
            if a.alive:
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy), radius, 1)
                # tint d'énergie : surimpression cercle plus petit
                e_frac = max(0.0, a.energy / env.cfg.max_energy)
                inner_r = max(1, int(radius * e_frac))
                if inner_r > 1:
                    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), inner_r, 1)

        # ─── HUD ───────────────────────────────────────────────────────────
        hud_y0 = env.cfg.rows * cell_px
        pygame.draw.rect(screen, HUD_PANEL, pygame.Rect(0, hud_y0, width, hud_h))

        alive_rate = env.n_alive / env.cfg.n_agents
        outcome_color = HUD_RED if last_outcome == "ALL DEAD" else (
            HUD_GREEN if last_outcome.startswith("TRUNCATED") else HUD_DIM
        )

        line1 = (
            f"policy={policies[policy_idx].name:7s}  N={env.cfg.n_agents:3d}  "
            f"alive={env.n_alive:3d}/{env.cfg.n_agents:<3d} ({alive_rate:5.1%})  "
            f"step={env.step_count:4d}/{env.cfg.max_steps}"
        )
        line2 = (
            f"food_in_grid={env.food_count:4d}  "
            f"dead={env.n_dead:3d}  episode#{episode_idx}  "
            f"delay={delay_ms}ms"
        )
        line3 = f"last: {last_outcome}  (final alive {last_alive_final})"
        controls = (
            "SPACE pause  R reset  A switch policy  N cycle density  "
            "↑/↓ speed  Q/ESC quit"
        )

        screen.blit(font_lg.render(line1, True, HUD_FG), (12, hud_y0 + 12))
        screen.blit(font_sm.render(line2, True, HUD_DIM), (12, hud_y0 + 38))
        screen.blit(font_sm.render(line3, True, outcome_color), (12, hud_y0 + 58))
        screen.blit(font_sm.render(controls, True, HUD_DIM), (12, hud_y0 + 80))

        if paused:
            screen.blit(font_lg.render("PAUSED", True, HUD_RED), (width - 90, hud_y0 + 12))

        pygame.display.flip()
        clock.tick(60)
        if delay_ms > 0 and not paused:
            pygame.time.wait(delay_ms)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_gui_v2()
