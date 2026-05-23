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

import colorsys
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


# ─── viz config (V5.4 — refonte lisibilité) ──────────────────────────────
TRAIL_LEN = 12
# V5.6 — flashes beaucoup plus longs pour visibilité
EAT_FLASH_FRAMES = 12
AGE_GRAY = (190, 190, 195)
MAX_AGE_MIX = 0.45
HALO_EXTRA = 4
BIRTH_FLASH_FRAMES = 45            # +125 % pour bien voir les naissances
BIRTH_FLASH_COLOR = (120, 200, 255)
POP_CURVE_SAMPLES = 200
POP_CURVE_H = 50
# V5 — construction (nid en maison stylisée)
BUILD_FLASH_FRAMES = 40            # +166 % pour bien voir les constructions
BUILD_FLASH_COLOR = (255, 200, 100)
# V5.6 — event log
EVENT_LOG_MAXLEN = 6
NEST_BASE_ALPHA = 240           # nids beaucoup plus opaques qu'avant
NEST_ROOF_ALPHA = 255
# V5.3 — cache viz
CACHE_BAR_COLOR = (90, 230, 90)
CACHE_BAR_EMPTY_COLOR = (40, 80, 40)
# food viz
FOOD_BG = (50, 110, 50)
FOOD_DOT = (180, 240, 110)
# heatmap moins agressive (alpha plus bas)
HEATMAP_ALPHA = 90              # plus discret (avant : 255 opaque)
# panneau légende
LEGEND_W = 220


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

_GOLDEN_RATIO_CONJ = 0.6180339887498949   # 1/phi


def agent_color(agent_id: int) -> tuple[int, int, int]:
    """Couleur distincte par agent_id via golden-angle HSV.

    Quel que soit le nombre d'agents, chaque ID a une couleur unique et
    bien espacée des voisines (rotation par 1/phi sur le cercle des teintes).
    """
    h = (agent_id * _GOLDEN_RATIO_CONJ) % 1.0
    # Vary saturation/value un peu pour différencier davantage
    s = 0.62 + (agent_id % 3) * 0.08          # 0.62, 0.70, 0.78
    v = 0.85 + ((agent_id // 5) % 2) * 0.10   # 0.85 ou 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choisit blanc ou noir selon la luminance du fond pour lisibilité."""
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (20, 20, 25) if luminance > 130 else (245, 245, 250)

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


def _smart_policy_factory(env: SeasonalMultiAgentFoodGrid, seed: int = 0):
    """V5.5 — politique heuristique stratégique (cherche food, va au nid, build, cache)."""
    from aetherlife.agents.smart_heuristic import SmartHeuristicAgent
    return SmartHeuristicAgent(env, seed=seed)


def run_gui_v3(
    base_cfg: SeasonalMultiAgentConfig | None = None,
    *,
    n_choices: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    cell_px: int = 24,           # plus grand par défaut (était 18)
    hud_h: int = 180,             # +30 px pour le panneau events
    tick_delay_ms: int = 100,    # plus lent (60 → 100) pour visibilité
    seed: int = 0,
    show_temp_heatmap: bool = False,   # désactivé par défaut (toggle T)
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
    pygame.display.set_caption("AetherLife V5.3 — Seasonal Sandbox (caches + tribes)")
    grid_w = base_cfg.cols * cell_px
    width = grid_w + LEGEND_W
    height = base_cfg.rows * cell_px + hud_h
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font_lg = pygame.font.SysFont("consolas", 19, bold=True)
    font_md = pygame.font.SysFont("consolas", 14, bold=True)
    font_sm = pygame.font.SysFont("consolas", 12)

    # V5.5 — choix policy par défaut : Smart si build/repro activés, sinon Random
    default_smart = (
        base_cfg.build.enabled or base_cfg.reproduction.enabled
        or base_cfg.cache.enabled
    )
    policy_slots = [
        ("Smart", lambda e: _smart_policy_factory(e, seed=seed)),
        ("Random", lambda e: _random_policy_factory(e, seed=seed)),
    ]
    policy_idx = 0 if default_smart else 1
    policy = policy_slots[policy_idx][1](env)
    show_temp = show_temp_heatmap
    paused = False
    delay_ms = tick_delay_ms
    episode_idx = 0
    last_outcome = "—"
    showing_report = False
    report_lines: list[str] = []
    tracker = EpisodeStatsTracker(n_agents=env.cfg.n_agents, track_seasons=True)
    tracker.reset(env)
    # --- V3.8 living sandbox state + V4 evolution viz ---
    trails: dict[int, deque] = {
        a.agent_id: deque(maxlen=TRAIL_LEN) for a in env._agents  # noqa: SLF001
    }
    eat_flash_frames: dict[int, int] = {}
    birth_flash_frames: dict[int, int] = {}     # V4 — flash bleu sur enfant
    build_flash_frames: dict[int, int] = {}     # V5 — flash sur nid neuf
    pop_curve: deque = deque(maxlen=POP_CURVE_SAMPLES)
    pop_curve.append(env.n_alive)
    show_trails = True
    show_nests = True
    # V5.6 — event log
    event_log: deque = deque(maxlen=EVENT_LOG_MAXLEN)

    def reset_episode(new_seed: int | None = None) -> None:
        nonlocal obs_dict, tracker, showing_report, trails, eat_flash_frames
        nonlocal birth_flash_frames, build_flash_frames, pop_curve
        s = new_seed if new_seed is not None else seed + episode_idx
        obs_dict, _ = env.reset(seed=s)
        tracker = EpisodeStatsTracker(n_agents=env.cfg.n_agents, track_seasons=True)
        tracker.reset(env)
        showing_report = False
        trails = {a.agent_id: deque(maxlen=TRAIL_LEN) for a in env._agents}  # noqa: SLF001
        eat_flash_frames = {}
        birth_flash_frames = {}
        build_flash_frames = {}
        pop_curve = deque(maxlen=POP_CURVE_SAMPLES)
        pop_curve.append(env.n_alive)

    def switch_density() -> None:
        nonlocal n_idx, env, policy, obs_dict, episode_idx, last_outcome
        n_idx = (n_idx + 1) % len(n_choices)
        env = make_env(n_choices[n_idx])
        policy = policy_slots[policy_idx][1](env)
        episode_idx += 1
        reset_episode()
        last_outcome = "—"

    def switch_policy() -> None:
        nonlocal policy_idx, policy, episode_idx, last_outcome
        policy_idx = (policy_idx + 1) % len(policy_slots)
        policy = policy_slots[policy_idx][1](env)
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
                elif event.key == pygame.K_a:
                    switch_policy()
                elif event.key == pygame.K_t:
                    show_temp = not show_temp
                elif event.key == pygame.K_l:
                    show_trails = not show_trails
                elif event.key == pygame.K_b:
                    show_nests = not show_nests
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
            # V4 — flash bleu sur les enfants nés ce tick
            for child_id in env.births_last_step:
                birth_flash_frames[child_id] = BIRTH_FLASH_FRAMES
                trails.setdefault(child_id, deque(maxlen=TRAIL_LEN))
                # V5.6 — log birth
                try:
                    parent = next(
                        (a for a in env._agents if a.agent_id == child_id),  # noqa: SLF001
                        None,
                    )
                    parent_id = parent.parent_id if parent else "?"
                except Exception:
                    parent_id = "?"
                event_log.append((
                    f"t{env.step_count}", "BIRTH", f"#{child_id}<-#{parent_id}",
                    BIRTH_FLASH_COLOR,
                ))
            # V5 — flash sur les nids fraîchement construits
            for nest in env.builds_last_step:
                build_flash_frames[nest.owner_id] = BUILD_FLASH_FRAMES
                event_log.append((
                    f"t{env.step_count}", "BUILD", f"#{nest.owner_id}@({nest.pos[0]},{nest.pos[1]})",
                    BUILD_FLASH_COLOR,
                ))
            # V5.6 — log deaths
            for aid in list(env.alive_agent_ids):
                pass
            # Detect deaths via change in n_alive
            current_dead_ids = {
                a.agent_id for a in env._agents if not a.alive  # noqa: SLF001
            }
            new_deaths = current_dead_ids - getattr(env, "_prev_dead_ids", set())
            env._prev_dead_ids = current_dead_ids  # noqa: SLF001
            for did in new_deaths:
                event_log.append((
                    f"t{env.step_count}", "DEATH ", f"#{did}",
                    (220, 100, 100),
                ))
            # V4 — sample pop curve
            pop_curve.append(env.n_alive)
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

        # 1) base cell color (heatmap optionnel, sinon uniforme)
        for r in range(env.cfg.rows):
            for c in range(env.cfg.cols):
                x, y = c * cell_px, r * cell_px
                rect = pygame.Rect(x, y, cell_px, cell_px)
                if show_temp:
                    base_color = _temp_to_color(float(temp_field[r, c]), tmin, tmax)
                    # rend la heatmap plus subtile
                    cf = CELL_FREE
                    mix = HEATMAP_ALPHA / 255
                    base_color = tuple(
                        int(cf[i] * (1 - mix) + base_color[i] * mix) for i in range(3)
                    )
                    pygame.draw.rect(screen, base_color, rect)
                else:
                    pygame.draw.rect(screen, CELL_FREE, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, 1)

        # 2) food : cercle bordé en vert avec point clair central (plus lisible)
        for r in range(env.cfg.rows):
            for c in range(env.cfg.cols):
                if not env.food_mask[r, c]:
                    continue
                x, y = c * cell_px, r * cell_px
                cxc = x + cell_px // 2
                cyc = y + cell_px // 2
                food_r = max(3, cell_px // 3)
                pygame.draw.circle(screen, FOOD_BG, (cxc, cyc), food_r)
                pygame.draw.circle(screen, CELL_FOOD, (cxc, cyc), food_r, 2)
                pygame.draw.circle(screen, FOOD_DOT, (cxc, cyc), max(1, food_r // 3))

        # 2bis) V6 — plants in growth : taille proportionnelle au stade
        plants_dict = getattr(env, "plants", {})
        if plants_dict:
            current_tick = env.step_count
            for (pr, pc), plant in plants_dict.items():
                prog = plant.progress(current_tick)
                x, y = pc * cell_px, pr * cell_px
                cxc = x + cell_px // 2
                cyc = y + cell_px // 2
                # Stem (tige) verte qui grandit
                stem_max_h = cell_px - 6
                stem_h = max(2, int(stem_max_h * prog))
                stem_color = (
                    int(100 + 60 * prog),  # plus mûr = plus jaune
                    int(180 + 50 * prog),
                    50,
                )
                pygame.draw.rect(
                    screen, stem_color,
                    pygame.Rect(cxc - 1, cyc + cell_px // 2 - 3 - stem_h, 2, stem_h),
                )
                # Tête plante : cercle qui grossit
                head_r = max(2, int((cell_px // 5) * prog))
                if head_r >= 2:
                    pygame.draw.circle(
                        screen, stem_color,
                        (cxc, cyc + cell_px // 2 - 3 - stem_h),
                        head_r,
                    )

        # 3) V5/V5.3 — draw nests as HOUSES (toit triangulaire + corps)
        if show_nests:
            cache_stock = getattr(env, "nest_food_stock", {})
            cache_capacity = (
                env.cfg.cache.max_capacity
                if hasattr(env.cfg, "cache") and env.cfg.cache.enabled
                else 1.0
            )
            for nest in env.nests.values():
                nr, nc = nest.pos
                nx = nc * cell_px
                ny = nr * cell_px
                nest_color = agent_color(nest.owner_id)
                roof_color = (
                    min(255, nest_color[0] + 30),
                    min(255, nest_color[1] + 30),
                    min(255, nest_color[2] + 30),
                )
                # Corps de la maison (carré inférieur)
                body_h = int(cell_px * 0.65)
                body_y = ny + cell_px - body_h - 1
                body_rect = pygame.Rect(nx + 2, body_y, cell_px - 4, body_h)
                pygame.draw.rect(screen, nest_color, body_rect)
                pygame.draw.rect(screen, (20, 20, 20), body_rect, 2)
                # Toit triangulaire
                roof_y_top = ny + 1
                roof_y_base = body_y
                roof_pts = [
                    (nx + 1, roof_y_base),
                    (nx + cell_px - 1, roof_y_base),
                    (nx + cell_px // 2, roof_y_top),
                ]
                pygame.draw.polygon(screen, roof_color, roof_pts)
                pygame.draw.polygon(screen, (20, 20, 20), roof_pts, 2)
                # Porte (rectangle sombre au centre du corps)
                door_w = max(2, cell_px // 6)
                door_h = max(3, body_h // 2)
                door_x = nx + cell_px // 2 - door_w // 2
                door_y = body_y + body_h - door_h - 1
                pygame.draw.rect(
                    screen, (30, 30, 35),
                    pygame.Rect(door_x, door_y, door_w, door_h),
                )
                # V5.5 — ID owner sur le toit (avec petit fond pour lisibilité)
                if cell_px >= 20:
                    owner_id_text = str(nest.owner_id)
                    txt_color = text_color_for(roof_color)
                    id_surf = font_sm.render(owner_id_text, True, txt_color)
                    id_rect = id_surf.get_rect(
                        center=(nx + cell_px // 2, roof_y_top + (roof_y_base - roof_y_top) // 2 + 1)
                    )
                    screen.blit(id_surf, id_rect)
                # V5.3 — barre cache à droite (silo)
                stock = cache_stock.get(nest.owner_id, 0)
                if cache_capacity > 0 and cache_capacity > 1:
                    fill_frac = min(1.0, stock / cache_capacity)
                    bar_w = max(3, cell_px // 6)
                    bar_h_max = body_h - 2
                    bar_x = nx + cell_px - bar_w - 1
                    bar_y_bottom = body_y + body_h - 1
                    # Silo vide
                    pygame.draw.rect(
                        screen, CACHE_BAR_EMPTY_COLOR,
                        pygame.Rect(bar_x, body_y, bar_w, bar_h_max),
                    )
                    # Remplissage
                    fill_h = int(bar_h_max * fill_frac)
                    if fill_h > 0:
                        pygame.draw.rect(
                            screen, CACHE_BAR_COLOR,
                            pygame.Rect(bar_x, bar_y_bottom - fill_h, bar_w, fill_h),
                        )
                    pygame.draw.rect(
                        screen, (20, 20, 20),
                        pygame.Rect(bar_x, body_y, bar_w, bar_h_max), 1,
                    )
                # Build flash (anneau qui s'étend autour de la maison)
                if build_flash_frames.get(nest.owner_id, 0) > 0:
                    bprog = 1.0 - (build_flash_frames[nest.owner_id] / BUILD_FLASH_FRAMES)
                    bflash_r = int(cell_px * (0.6 + 0.6 * bprog))
                    bflash_alpha = int(220 * (1 - bprog))
                    bfsurf = pygame.Surface(
                        (bflash_r * 2 + 2, bflash_r * 2 + 2), pygame.SRCALPHA
                    )
                    pygame.draw.circle(
                        bfsurf, (*BUILD_FLASH_COLOR, bflash_alpha),
                        (bflash_r + 1, bflash_r + 1), bflash_r, 3,
                    )
                    screen.blit(
                        bfsurf,
                        (nx + cell_px // 2 - bflash_r - 1,
                         ny + cell_px // 2 - bflash_r - 1),
                    )
                    build_flash_frames[nest.owner_id] -= 1

        # V3.8 — draw trails first (under agents)
        if show_trails:
            for aid, trail in trails.items():
                base_color = agent_color(aid)
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
                # croix rouge sur les morts (plus lisible)
                rr = max(2, radius - 1)
                pygame.draw.line(
                    screen, (200, 80, 80),
                    (cx - rr, cy - rr), (cx + rr, cy + rr), 2,
                )
                pygame.draw.line(
                    screen, (200, 80, 80),
                    (cx - rr, cy + rr), (cx + rr, cy - rr), 2,
                )
                continue
            base_color = agent_color(a.agent_id)
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
            # V4 — Birth flash bleu (l'enfant brille à sa naissance)
            if birth_flash_frames.get(a.agent_id, 0) > 0:
                bprogress = 1.0 - (birth_flash_frames[a.agent_id] / BIRTH_FLASH_FRAMES)
                bflash_r = radius + int(3 + 14 * bprogress)
                bflash_alpha = int(200 * (1 - bprogress))
                bsurf = pygame.Surface(
                    (bflash_r * 2 + 2, bflash_r * 2 + 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    bsurf, (*BIRTH_FLASH_COLOR, bflash_alpha),
                    (bflash_r + 1, bflash_r + 1), bflash_r, 3,
                )
                screen.blit(bsurf, (cx - bflash_r - 1, cy - bflash_r - 1))
                birth_flash_frames[a.agent_id] -= 1
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
            # V5.5 — ID de l'agent affiché si cellule assez grande
            if cell_px >= 20:
                txt_color = text_color_for(color)
                id_surf = font_md.render(str(a.agent_id), True, txt_color)
                id_rect = id_surf.get_rect(center=(cx, cy))
                screen.blit(id_surf, id_rect)
            # V6.1 — petits points verts au-dessus pour les graines
            seeds = getattr(a, "seeds", 0)
            if seeds > 0 and cell_px >= 20:
                seed_dot_r = max(2, cell_px // 14)
                spacing = seed_dot_r * 2 + 1
                total_w = min(seeds, 5) * spacing
                start_x = cx - total_w // 2
                seed_y = cy - radius - seed_dot_r - 2
                for si in range(min(seeds, 5)):
                    pygame.draw.circle(
                        screen, (130, 220, 90),
                        (start_x + si * spacing, seed_y), seed_dot_r,
                    )
                if seeds > 5:
                    extra_surf = font_sm.render(f"+{seeds-5}", True, (130, 220, 90))
                    screen.blit(extra_surf,
                                (start_x + 5 * spacing + 2, seed_y - 5))

        # V6.1 — winter approach warning (clignote en automne tard)
        if hasattr(env, "phase"):
            phase = env.phase
            if 0.6 <= phase < 0.75:
                warn_alpha = 180 + int(60 * abs(np.sin(env.step_count * 0.3)))
                warn_surf = pygame.Surface(
                    (grid_w, 24), pygame.SRCALPHA,
                )
                warn_surf.fill((200, 100, 50, 70))
                screen.blit(warn_surf, (0, 0))
                warn_text = font_md.render(
                    f"!! WINTER COMING (phase {phase:.2f})  STOCKEZ DES GRAINES !!",
                    True, (255, 220, 150),
                )
                screen.blit(warn_text, (12, 4))

        # ─── Légende latérale (à droite de la grille) ────────────────────
        legend_x0 = grid_w + 12
        legend_panel = pygame.Rect(grid_w, 0, LEGEND_W, env.cfg.rows * cell_px)
        pygame.draw.rect(screen, HUD_PANEL, legend_panel)
        pygame.draw.line(screen, GRID_LINE, (grid_w, 0),
                         (grid_w, env.cfg.rows * cell_px), 1)
        ly = 14
        screen.blit(font_lg.render("Legend", True, HUD_FG), (legend_x0, ly))
        ly += 26
        # food icon
        pygame.draw.circle(screen, FOOD_BG, (legend_x0 + 12, ly + 6), 7)
        pygame.draw.circle(screen, CELL_FOOD, (legend_x0 + 12, ly + 6), 7, 2)
        pygame.draw.circle(screen, FOOD_DOT, (legend_x0 + 12, ly + 6), 3)
        screen.blit(font_sm.render("food", True, HUD_FG), (legend_x0 + 28, ly + 1))
        ly += 22
        # nest icon (mini maison)
        nest_color_demo = agent_color(0)
        pygame.draw.rect(screen, nest_color_demo,
                         pygame.Rect(legend_x0 + 4, ly + 6, 16, 10))
        pygame.draw.rect(screen, (20, 20, 20),
                         pygame.Rect(legend_x0 + 4, ly + 6, 16, 10), 1)
        pygame.draw.polygon(screen, (255, 215, 100),
                            [(legend_x0 + 4, ly + 6),
                             (legend_x0 + 20, ly + 6),
                             (legend_x0 + 12, ly - 1)])
        pygame.draw.polygon(screen, (20, 20, 20),
                            [(legend_x0 + 4, ly + 6),
                             (legend_x0 + 20, ly + 6),
                             (legend_x0 + 12, ly - 1)], 1)
        screen.blit(font_sm.render("nest (couleur=owner)", True, HUD_FG),
                    (legend_x0 + 28, ly + 1))
        ly += 24
        # cache silo
        pygame.draw.rect(screen, CACHE_BAR_EMPTY_COLOR,
                         pygame.Rect(legend_x0 + 4, ly + 1, 5, 14))
        pygame.draw.rect(screen, CACHE_BAR_COLOR,
                         pygame.Rect(legend_x0 + 4, ly + 6, 5, 9))
        pygame.draw.rect(screen, (20, 20, 20),
                         pygame.Rect(legend_x0 + 4, ly + 1, 5, 14), 1)
        screen.blit(font_sm.render("silo cache (V5.3)", True, HUD_FG),
                    (legend_x0 + 18, ly + 1))
        ly += 22
        # agent vivant
        pygame.draw.circle(screen, agent_color(1), (legend_x0 + 12, ly + 6), 7)
        pygame.draw.circle(screen, (255, 255, 255), (legend_x0 + 12, ly + 6), 7, 1)
        screen.blit(font_sm.render("agent vivant", True, HUD_FG),
                    (legend_x0 + 28, ly + 1))
        ly += 22
        # agent mort
        pygame.draw.circle(screen, AGENT_DEAD, (legend_x0 + 12, ly + 6), 7)
        pygame.draw.line(screen, (200, 80, 80),
                         (legend_x0 + 6, ly), (legend_x0 + 18, ly + 12), 2)
        screen.blit(font_sm.render("agent mort", True, HUD_FG),
                    (legend_x0 + 28, ly + 1))
        ly += 22
        # birth flash
        pygame.draw.circle(screen, BIRTH_FLASH_COLOR, (legend_x0 + 12, ly + 6), 7, 2)
        screen.blit(font_sm.render("naissance (flash bleu)", True, HUD_FG),
                    (legend_x0 + 28, ly + 1))
        ly += 22
        # eat flash
        pygame.draw.circle(screen, (255, 255, 200), (legend_x0 + 12, ly + 6), 7, 2)
        screen.blit(font_sm.render("eat (flash jaune)", True, HUD_FG),
                    (legend_x0 + 28, ly + 1))
        ly += 22
        # build flash
        pygame.draw.circle(screen, BUILD_FLASH_COLOR, (legend_x0 + 12, ly + 6), 7, 2)
        screen.blit(font_sm.render("construit (flash orange)", True, HUD_FG),
                    (legend_x0 + 28, ly + 1))
        ly += 28
        # Modes activés (live status)
        screen.blit(font_md.render("Modes actifs :", True, HUD_FG), (legend_x0, ly))
        ly += 18
        rcfg = env.cfg.reproduction
        bcfg = env.cfg.build
        ccfg = env.cfg.cache
        pcfg = getattr(env.cfg, "planting", None)
        plant_on = pcfg.enabled if pcfg else False
        for label, active in [
            ("Reproduction", rcfg.enabled),
            ("Construction", bcfg.enabled),
            ("Family inherit.", bcfg.family_inheritance),
            ("Cache food", ccfg.enabled),
            ("Plantation V6", plant_on),
            ("Heatmap temp.", show_temp),
            ("Trails", show_trails),
        ]:
            color = HUD_GREEN if active else HUD_DIM
            mark = "[ON]" if active else "[off]"
            screen.blit(font_sm.render(f"{mark:5s} {label}", True, color),
                        (legend_x0 + 4, ly))
            ly += 15

        # ─── HUD bas ──────────────────────────────────────────────────────
        hud_y0 = env.cfg.rows * cell_px
        pygame.draw.rect(screen, HUD_PANEL, pygame.Rect(0, hud_y0, width, hud_h))

        season_name = SEASON_LABELS[int(env.season)]
        season_tint = SEASON_TINTS[env.season]
        denom_pop = max(env.cfg.n_agents, 1)
        alive_rate = env.n_alive / denom_pop
        outcome_color = HUD_RED if last_outcome == "ALL DEAD" else (
            HUD_GREEN if last_outcome.startswith("TRUNCATED") else HUD_DIM
        )
        period = env.cfg.seasonal.season_period
        step_in_season = env.step_count % period
        ticks_to_next = period // 4 - (step_in_season % (period // 4))
        total_pop = len(env._agents)  # noqa: SLF001 — incl. dead

        line1 = (
            f"season={season_name:6s}  phase={env.phase:5.2f}  "
            f"next-in={ticks_to_next:3d}t  step={env.step_count:4d}/{env.cfg.max_steps}"
        )
        n_alive_disp = env.n_alive
        n_nests = env.n_nests
        cache_total = getattr(env, "total_cached_food", 0.0)
        n_plants = getattr(env, "n_plants", 0)
        n_matured = getattr(env, "plants_matured_total", 0)
        # V6.1 — total graines + indicateur winter approach
        total_seeds = sum(
            getattr(a, "seeds", 0) for a in env._agents if a.alive  # noqa: SLF001
        )
        line2 = (
            f"alive={n_alive_disp:3d}/{total_pop:<3d}  births={env.n_births_total:3d}  "
            f"nests={n_nests:3d}  plants={n_plants:3d}(matured={n_matured:3d})  "
            f"seeds={total_seeds:3d}  cache={cache_total:4.0f}  food={env.food_count:4d}"
        )
        line3 = f"last: {last_outcome}"
        line4 = "Mean food regen factors per season:"
        line5 = (
            f"  spring={env.cfg.seasonal.spring_lambda_factor:.2f}  "
            f"summer={env.cfg.seasonal.summer_lambda_factor:.2f}  "
            f"autumn={env.cfg.seasonal.autumn_lambda_factor:.2f}  "
            f"winter={env.cfg.seasonal.winter_lambda_factor:.2f}"
        )
        policy_name = policy_slots[policy_idx][0]
        controls = (
            f"SPACE pause/next  R reset  A policy ({policy_name})  N density  "
            f"T heatmap  L trails  B nests  ↑/↓ speed  Q"
        )

        screen.blit(font_lg.render(line1, True, season_tint), (12, hud_y0 + 8))
        screen.blit(font_sm.render(line2, True, HUD_FG), (12, hud_y0 + 32))
        screen.blit(font_sm.render(line3, True, outcome_color), (12, hud_y0 + 50))
        screen.blit(font_sm.render(line5, True, HUD_DIM), (12, hud_y0 + 80))
        screen.blit(font_sm.render(controls, True, HUD_DIM), (12, hud_y0 + 110))

        # V5.6 — Event log panel (right side of HUD)
        event_x0 = grid_w - 280
        screen.blit(font_md.render("Recent events:", True, HUD_FG),
                    (event_x0, hud_y0 + 8))
        for i, (ts, kind, detail, color) in enumerate(list(event_log)):
            y = hud_y0 + 28 + i * 16
            screen.blit(
                font_sm.render(f"{ts:6s} {kind:6s} {detail}", True, color),
                (event_x0, y),
            )

        # V4 — courbe population mini-graph en haut à gauche du grid
        if len(pop_curve) > 1:
            curve_w = min(180, grid_w - 24)
            curve_x0 = 10
            curve_y0 = 10
            pts = list(pop_curve)
            max_pop_observed = max(max(pts), env.cfg.n_agents)
            # fond translucide
            curve_bg = pygame.Surface((curve_w + 8, POP_CURVE_H + 8), pygame.SRCALPHA)
            curve_bg.fill((10, 10, 14, 180))
            pygame.draw.rect(curve_bg, (90, 200, 90), curve_bg.get_rect(), 1)
            screen.blit(curve_bg, (curve_x0 - 4, curve_y0 - 4))
            # points
            for i in range(1, len(pts)):
                x1 = curve_x0 + int((i - 1) * curve_w / (len(pts) - 1))
                x2 = curve_x0 + int(i * curve_w / (len(pts) - 1))
                y1 = curve_y0 + POP_CURVE_H - int(pts[i - 1] / max_pop_observed * POP_CURVE_H)
                y2 = curve_y0 + POP_CURVE_H - int(pts[i] / max_pop_observed * POP_CURVE_H)
                pygame.draw.line(screen, (90, 200, 90), (x1, y1), (x2, y2), 2)
            label = font_sm.render(
                f"pop {pts[-1]}/{max_pop_observed}", True, HUD_DIM
            )
            screen.blit(label, (curve_x0, curve_y0 + POP_CURVE_H + 2))

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
