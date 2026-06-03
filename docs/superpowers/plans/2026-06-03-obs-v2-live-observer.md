# OBS Viewer 2.0 (lite) — Live V8 Observer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un GUI live qui fait tourner la config V8 (coordination), se règle en jours, et affiche le compte rendu de l'Historien d'une touche.

**Architecture:** Réutilise le rendu V8 (`pygame_viewer_v8._draw_frame`) et la sim (`build_env`+`LineageAgent`, overnight non modifié). Nouveau : `build_live_report` (assemble un report dict pour l'Historien depuis l'env live), la boucle `live_viewer_v8` (step + render + clavier + overlay Historien), et l'entrée CLI.

**Tech Stack:** Python 3.13, pygame-ce, torch (cerveaux), numpy, pytest. `build_live_report` testable CPU ; boucle = smoke headless (SDL dummy).

**Spec :** `docs/superpowers/specs/2026-06-03-obs-viewer2-live-v8-observer-design.md`

**API vérifiée :**
- env (`SeasonalMultiAgentFoodGrid`) : `.n_alive`, `.n_births_total`, `.gather_successes_total`, `.gather_failures_total`, `.season`, `.step_count`, `._agents` (liste `_AgentState`: `agent_id,pos,energy,alive,root_ancestor_id,biome_affinity`), `.gather_spots`, `._tokens_this_tick`, `.coop_metrics.finalize()`, `.cfg.{rows,cols,max_energy,vocabulary.n_tokens,vocabulary.enabled,cooperative.enabled}`.
- vocab d'un brain : `b.vocabulary.usage_count` (np int array n_tokens), `b.vocabulary.usage_entropy()`, `b.vocabulary.distance_to(other)`, `b.root_id`.
- `policy.registry` : itérable de `LineageBrain`, `policy.registry.total_global_steps()`.
- `build_env(seed, regime=, vocalize_energy_cost=)`, `LineageAgent(env=, cfg=, n_actions=4, seed=)`, `policy.act_dict(obs_stub, greedy=False)`.
- `historian/spatial_mobility` : `OccupancyAccumulator(rows,cols)`, `window_bounds(total)`, `build_spatial_mobility_block(s,e,start_window=,end_window=)`.
- `Historian.from_report(report_dict)` → `.discoveries` (list `Discovery(slug,category,confidence,headline)`), `.render_summary()`, `.write_all(out_dir)`.
- `pygame_viewer_v8._draw_frame(event, meta, cell_px, focus_lineage=None)` → `pygame.Surface`. Event schema : `{t,season,n_alive,n_lin,agents:[{id,lin,r,c,e}],vocal:{sid:tok},spots:[{r,c,n}]}`.

---

## File Structure

| Fichier | Responsabilité |
|---|---|
| `aetherlife/viz/live_report.py` | `build_live_report` (+ helper `_language_metrics`) → report dict Historien |
| `aetherlife/viz/live_viewer_v8.py` | `build_event_dict`, `render_live_frame`, `run_live` (boucle + clavier + overlay) |
| `scripts/launch_gui_v8.py` | entrée CLI |
| `tests/test_live_report.py` | assemblage + Historian (CPU) |
| `tests/test_live_viewer_v8.py` | event dict + render + smoke loop (headless) |

---

## Task 1: `build_live_report` (assemblage report Historien)

**Files:**
- Create: `aetherlife/viz/live_report.py`
- Test: `tests/test_live_report.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_live_report.py` :

```python
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)


def _tiny_env_policy():
    from overnight_v8b1 import build_env
    from aetherlife.agents.lineage_agent import LineageAgent
    from aetherlife.agents.lineage_brain import BrainConfig
    import numpy as np
    env = build_env(1, regime="coordination_collective", vocalize_energy_cost=0.05)
    cfg = BrainConfig(enabled=True, device="cpu", vision_radius=4,
                      hidden_dims=(64, 64))
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=1)
    for _ in range(30):
        if env.n_alive == 0:
            break
        obs_stub = {a.agent_id: np.zeros(10)
                    for a in env._agents if a.alive}  # noqa: SLF001
        env.step(policy.act_dict(obs_stub, greedy=False))
    return env, policy


def test_build_live_report_has_blocks():
    from aetherlife.viz.live_report import build_live_report
    from aetherlife.historian.spatial_mobility import OccupancyAccumulator
    env, policy = _tiny_env_policy()
    occ = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    occ.add_positions([(a.pos[0], a.pos[1])
                       for a in env._agents if a.alive])  # noqa: SLF001
    rep = build_live_report(env, policy, occ, occ,
                            windows=((0, 10), (20, 30)), n_ticks=30)
    for k in ("final_state", "criterion_3_selection", "language_metrics_v8b2",
              "cooperative_v8c3", "cooperative_metrics_v8c3",
              "spatial_mobility_v8c3", "config"):
        assert k in rep, f"bloc manquant : {k}"
    assert rep["final_state"]["n_alive"] == env.n_alive
    assert rep["criterion_3_selection"]["n_lineages_final"] == len(policy.registry)


def test_build_live_report_feeds_historian():
    from aetherlife.viz.live_report import build_live_report
    from aetherlife.historian import Historian
    from aetherlife.historian.spatial_mobility import OccupancyAccumulator
    env, policy = _tiny_env_policy()
    occ = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    occ.add_positions([(a.pos[0], a.pos[1])
                       for a in env._agents if a.alive])  # noqa: SLF001
    rep = build_live_report(env, policy, occ, occ,
                            windows=((0, 10), (20, 30)), n_ticks=30)
    h = Historian.from_report(rep, run_id="live_test")
    assert isinstance(h.discoveries, list)  # pas de crash, liste (≥0)
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_live_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aetherlife.viz.live_report'`

- [ ] **Step 3: Implémenter build_live_report**

Créer `aetherlife/viz/live_report.py` :

```python
"""OBS Viewer 2.0 — assemble un report dict Historien depuis un env V8 LIVE.

Réutilise les builders existants (coop_metrics.finalize, spatial_mobility) et
reproduit le petit calcul language_metrics de overnight_v8b1 (sans modifier le
runner taggé). Le DiscoveriesDetector retourne [] pour les blocs absents.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from aetherlife.historian.spatial_mobility import build_spatial_mobility_block


def _language_metrics(env, policy) -> dict[str, Any]:
    """Reproduit overnight_v8b1 §language_metrics depuis le registry vocab."""
    if not env.cfg.vocabulary.enabled:
        return {}
    brains = [b for b in policy.registry if b.vocabulary is not None]
    if not brains:
        return {}
    n_tokens = env.cfg.vocabulary.n_tokens
    total = sum(int(b.vocabulary.usage_count.sum()) for b in brains)
    concentrations = []
    for tok in range(n_tokens):
        per = {b.root_id: int(b.vocabulary.usage_count[tok]) for b in brains}
        s = sum(per.values())
        if s > 0:
            concentrations.append(max(per.values()) / s)
    mean_conc = float(np.mean(concentrations)) if concentrations else 0.0
    entropies = [b.vocabulary.usage_entropy() for b in brains]
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    distances = [
        brains[i].vocabulary.distance_to(brains[j].vocabulary)
        for i in range(len(brains)) for j in range(i + 1, len(brains))
    ]
    mean_dist = float(np.mean(distances)) if distances else 0.0
    per_token_top = {
        str(t): int(sum(b.vocabulary.usage_count[t] for b in brains))
        for t in range(n_tokens)
    }
    return {
        "n_brains_with_vocab": len(brains),
        "total_vocalize_count": total,
        "entropy_ratio": mean_entropy / max(float(np.log(n_tokens)), 1e-9),
        "mean_usage_entropy": mean_entropy,
        "mean_token_lineage_concentration": mean_conc,
        "mean_inter_lineage_distance": mean_dist,
        "per_token_usage_top": per_token_top,
    }


def build_live_report(env, policy, occ_start, occ_end, *,
                      windows: tuple, n_ticks: int) -> dict[str, Any]:
    """Report dict consommable par Historian/DiscoveriesDetector (live, MVP)."""
    alive = [a for a in env._agents if a.alive]  # noqa: SLF001
    lin_counts = Counter(a.root_ancestor_id for a in alive)
    n_alive = env.n_alive
    top = [
        {"root_id": r, "alive": c, "pct": 100 * c / max(n_alive, 1)}
        for r, c in lin_counts.most_common(5)
    ]
    aff = Counter(a.biome_affinity for a in alive)
    n_founders = env.cfg.n_agents
    swin, ewin = windows
    return {
        "config": {"seed": None, "n_ticks": n_ticks, "device": "live",
                   "vision_radius": None},
        "final_state": {
            "n_alive": n_alive,
            "n_births_total": env.n_births_total,
            "n_deaths": env.n_births_total + n_founders - n_alive,
            "top_lineages": top,
            "affinity_distribution": {str(k): v for k, v in aff.items()},
            "n_affinities_alive": len(aff),
        },
        "criterion_3_selection": {
            "n_lineages_initial": n_founders,
            "n_lineages_final": len(policy.registry),
            "dominant_lineage_pct": (top[0]["pct"] if top else 0.0),
        },
        "language_metrics_v8b2": _language_metrics(env, policy),
        "cooperative_v8c3": {
            "enabled": bool(env.cfg.cooperative.enabled),
            "gather_successes_total": int(env.gather_successes_total),
            "gather_failures_total": int(env.gather_failures_total),
        },
        "cooperative_metrics_v8c3": (
            env.coop_metrics.finalize() if env.cfg.cooperative.enabled else {}
        ),
        "spatial_mobility_v8c3": build_spatial_mobility_block(
            occ_start, occ_end, start_window=swin, end_window=ewin,
        ),
    }
```

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_live_report.py -q`
Expected: PASS (2 tests). Construit un env coordination CPU + 30 ticks (quelques s).

- [ ] **Step 5: Commit**

```bash
git add aetherlife/viz/live_report.py tests/test_live_report.py
git commit -m "feat(obs-v2): build_live_report — report Historien depuis env V8 live"
```

---

## Task 2: `live_viewer_v8` — event dict + render + boucle

**Files:**
- Create: `aetherlife/viz/live_viewer_v8.py`
- Test: `tests/test_live_viewer_v8.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_live_viewer_v8.py` :

```python
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np


def _env_policy():
    import sys
    sys.path.insert(0, "scripts")
    from overnight_v8b1 import build_env
    from aetherlife.agents.lineage_agent import LineageAgent
    from aetherlife.agents.lineage_brain import BrainConfig
    env = build_env(1, regime="coordination_collective", vocalize_energy_cost=0.05)
    cfg = BrainConfig(enabled=True, device="cpu", vision_radius=4,
                      hidden_dims=(64, 64))
    policy = LineageAgent(env=env, cfg=cfg, n_actions=4, seed=1)
    return env, policy


def test_build_event_dict_schema():
    from aetherlife.viz.live_viewer_v8 import build_event_dict
    env, policy = _env_policy()
    obs = {a.agent_id: np.zeros(10) for a in env._agents if a.alive}  # noqa: SLF001
    env.step(policy.act_dict(obs, greedy=False))
    ev = build_event_dict(env, policy, t=1)
    assert {"t", "season", "n_alive", "n_lin", "agents", "vocal", "spots"} <= set(ev)
    if ev["agents"]:
        assert {"id", "lin", "r", "c", "e"} <= set(ev["agents"][0])


def test_render_live_frame_surface():
    from aetherlife.viz.live_viewer_v8 import build_event_dict, render_live_frame
    env, policy = _env_policy()
    ev = build_event_dict(env, policy, t=1)
    meta = {"rows": env.cfg.rows, "cols": env.cfg.cols,
            "n_tokens": env.cfg.vocabulary.n_tokens}
    surf = render_live_frame(ev, meta, hud_extra="day 1/5  pop=20", cell_px=8)
    assert surf.get_width() == env.cfg.cols * 8
    assert surf.get_height() > env.cfg.rows * 8  # frame + HUD


def test_run_live_smoke_bounded():
    from aetherlife.viz.live_viewer_v8 import run_live
    # boucle bornée (max_frames) headless, ne doit pas crasher
    run_live(seed=1, regime="coordination_collective", device="cpu",
             days=1, ticks_per_day=20, cell_px=6, max_frames=15)
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_live_viewer_v8.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aetherlife.viz.live_viewer_v8'`

- [ ] **Step 3: Implémenter le viewer live**

Créer `aetherlife/viz/live_viewer_v8.py` :

```python
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
    pygame.quit()
```

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_live_viewer_v8.py -q`
Expected: PASS (3 tests). `run_live` smoke s'arrête à max_frames=15 (headless SDL dummy).

- [ ] **Step 5: Commit**

```bash
git add aetherlife/viz/live_viewer_v8.py tests/test_live_viewer_v8.py
git commit -m "feat(obs-v2): live_viewer_v8 — boucle live V8 + clavier + overlay Historien"
```

---

## Task 3: `launch_gui_v8.py` (entrée CLI)

**Files:**
- Create: `scripts/launch_gui_v8.py`
- Test: `tests/test_live_viewer_v8.py`

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_live_viewer_v8.py` :

```python
def test_launch_gui_v8_smoke(tmp_path):
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "scripts/launch_gui_v8.py", "--seed", "1",
         "--device", "cpu", "--days", "1", "--ticks-per-day", "20",
         "--max-frames", "12"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy",
             "PYTHONIOENCODING": "utf-8"},
    )
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_live_viewer_v8.py::test_launch_gui_v8_smoke -q`
Expected: FAIL — returncode != 0 (`scripts/launch_gui_v8.py` absent)

- [ ] **Step 3: Implémenter l'entrée**

Créer `scripts/launch_gui_v8.py` :

```python
"""OBS Viewer 2.0 (lite) — lance l'observateur LIVE V8.

Usage:
    python scripts/launch_gui_v8.py --days 5 --ticks-per-day 1000 --device cuda
Touches : ESPACE pause · +/- jours · H Historien · E export · ↑/↓ vitesse · ESC.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aetherlife.viz.live_viewer_v8 import run_live  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--regime", default="coordination_collective")
    p.add_argument("--device", default="cuda")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--ticks-per-day", type=int, default=1000)
    p.add_argument("--cell-px", type=int, default=14)
    p.add_argument("--max-frames", type=int, default=None,
                   help="borne pour smoke/test (None = illimité).")
    a = p.parse_args()
    run_live(seed=a.seed, regime=a.regime, device=a.device, days=a.days,
             ticks_per_day=a.ticks_per_day, cell_px=a.cell_px,
             max_frames=a.max_frames)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `pytest tests/test_live_viewer_v8.py::test_launch_gui_v8_smoke -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/launch_gui_v8.py tests/test_live_viewer_v8.py
git commit -m "feat(obs-v2): launch_gui_v8 — entree CLI live observer"
```

---

## Task 4: Vérification finale + lancement réel

**Files:** aucun nouveau (validation).

- [ ] **Step 1: Suite complète verte**

Run: `PYTHONIOENCODING=utf-8 pytest -q`
Expected: tous verts (existants + OBS V2), zéro régression.

- [ ] **Step 2: Lancement réel (GPU, fenêtre interactive)**

Note : un GUI pygame interactif a besoin d'un vrai display. Dans cette session
(SDL dummy / headless), il tourne mais sans fenêtre visible. À LANCER PAR
L'UTILISATEUR dans son terminal :
```
python scripts/launch_gui_v8.py --days 5 --ticks-per-day 1000 --device cuda
```
Vérifier : monde V8 qui vit (agents par lignée, halos tokens, gather spots),
HUD « day X/5 », touche H → overlay Historien (découvertes), +/- change les jours,
E exporte dans `results/gui_run/report/`.

- [ ] **Step 3: Commit doc (si ajustements)**

Aucun si tout marche ; sinon corriger + commit.

---

## Self-Review (auteur du plan)

- **Couverture spec** : build_live_report 6 blocs + language_metrics reproduit (T1) ✓ · réutilise _draw_frame (T2 render_live_frame) ✓ · boucle live + clavier ESPACE/+/-/H/E/↑↓/ESC (T2 run_live) ✓ · jours = ticks_per_day, +/- live (T2) ✓ · Niveau 1 overlay (T2 _historian_lines) + Niveau 2 write_all (T2 touche E) ✓ · entrée CLI --days --ticks-per-day (T3) ✓ · overnight non modifié (build_env réutilisé) ✓ · tests sans GPU (T1 CPU, T2 SDL dummy) ✓.
- **Écarts assumés** : `mobility_score` live n'est défini que si le budget a deux fenêtres remplies (sinon corr None — affiché tel quel, pas un bug). Détecteur héritage cognitif omis (criterion_1 absent → detect_cognition retourne []). `config.seed=None` dans le report (le GUI ne re-seed pas comme overnight) — sans impact sur les détecteurs.
- **Pas de placeholder** : code complet.
- **Cohérence types** : `build_live_report(env,policy,occ_s,occ_e,*,windows,n_ticks)->dict` identique T1/T2 ; `build_event_dict(env,policy,t)->dict` et `render_live_frame(ev,meta,*,hud_extra,cell_px)->Surface` identiques T2/tests ; event schema `{t,season,n_alive,n_lin,agents[id,lin,r,c,e],vocal,spots}` = ce que `_draw_frame` consomme (Viewer 1).
