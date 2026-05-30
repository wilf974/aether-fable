# V8 Replay Viewer (Viewer 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire un viewer replay/export pour le monde V8-C3 : re-simuler un seed → `events.jsonl` → rendre un clip MP4/GIF/PNG montrant lignées, tokens vocalisés et gather collectif, sans relancer le training.

**Architecture:** Découplage simulation/rendu via le contrat `events.jsonl`. Le recorder (GPU, réutilise `build_env` sans modifier le runner taggé) écrit le contrat ; le renderer (pur, testable sans GPU, SDL dummy) le lit et produit les frames. Helpers de couleurs/chargement isolés dans un module pur.

**Tech Stack:** Python 3.13, pygame-ce (rendu + PNG), imageio + imageio-ffmpeg (MP4), Pillow (GIF), numpy. Tests pytest sans GPU pour le renderer.

**Spec de référence:** `docs/superpowers/specs/2026-05-30-v8-replay-viewer-design.md`

---

## File Structure

| Fichier | Responsabilité |
|---|---|
| `aetherlife/viz/v8_replay.py` | **PUR** (pas de pygame) : couleurs token/lignée, chargement meta/events, validation |
| `aetherlife/viz/pygame_viewer_v8.py` | Rendu : `_draw_frame` (1 frame → Surface), `render_events` (events → PNG/GIF/MP4) |
| `scripts/render_v8.py` | CLI : `--events --out --fps --from-tick --to-tick --focus-lineage` |
| `scripts/record_events_v8.py` | Recorder GPU : re-sim un seed → `events.jsonl` + `meta.json` |
| `tests/test_v8_replay.py` | Tests helpers purs (couleurs, chargement) |
| `tests/test_pygame_viewer_v8.py` | Tests renderer sans GPU (pixel tint, halo token, frames, mp4) |
| `pyproject.toml` | Extra optionnel `viz` |

---

## Task 1: Dépendances — extra `viz`

**Files:**
- Modify: `pyproject.toml:16-20`

- [ ] **Step 1: Ajouter l'extra `viz`**

Dans `pyproject.toml`, sous `[project.optional-dependencies]`, après le bloc `dev`, ajouter :

```toml
viz = [
    "pygame-ce>=2.5",
    "imageio>=2.34",
    "imageio-ffmpeg>=0.4",
    "pillow>=10.0",
]
```
(Pillow est requis par imageio pour l'écriture GIF ; imageio-ffmpeg bundle ffmpeg pour le MP4.)

- [ ] **Step 2: Installer**

Run: `source .venv/Scripts/activate && pip install -e ".[viz]"`
Expected: installe imageio, imageio-ffmpeg, pillow (pygame-ce déjà présent). Termine par `Successfully installed ...`.

- [ ] **Step 3: Vérifier l'import**

Run: `python -c "import pygame, imageio, imageio_ffmpeg, PIL; print('viz OK')"`
Expected: `viz OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(viz): extra optionnel viz (pygame/imageio/ffmpeg/pillow)"
```

---

## Task 2: Helpers couleurs (module pur)

**Files:**
- Create: `aetherlife/viz/v8_replay.py`
- Test: `tests/test_v8_replay.py`

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/test_v8_replay.py` :

```python
from aetherlife.viz.v8_replay import token_color, lineage_color, TOKEN_COLORS


def test_token_color_known_and_wraps():
    assert token_color(0) == TOKEN_COLORS[0]
    assert token_color(1) == TOKEN_COLORS[1]
    # wrap modulo nb de tokens
    assert token_color(len(TOKEN_COLORS)) == TOKEN_COLORS[0]


def test_token_color_is_rgb_triplet():
    c = token_color(2)
    assert isinstance(c, tuple) and len(c) == 3
    assert all(0 <= v <= 255 for v in c)


def test_lineage_color_deterministic():
    assert lineage_color(12) == lineage_color(12)


def test_lineage_color_distinct_for_different_roots():
    assert lineage_color(1) != lineage_color(2)


def test_lineage_color_is_rgb_triplet():
    c = lineage_color(7)
    assert isinstance(c, tuple) and len(c) == 3
    assert all(0 <= v <= 255 for v in c)
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `pytest tests/test_v8_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aetherlife.viz.v8_replay'`

- [ ] **Step 3: Implémenter les couleurs**

Créer `aetherlife/viz/v8_replay.py` :

```python
"""Helpers purs (sans pygame) pour le V8 Replay Viewer.

Contrat events.jsonl + couleurs déterministes token/lignée.
"""
from __future__ import annotations

import colorsys
import json
from typing import Any, Iterator

SCHEMA_VERSION = 1

# 4 couleurs saturées distinctes (rouge, bleu, vert, jaune) — 1 par token.
TOKEN_COLORS: list[tuple[int, int, int]] = [
    (231, 76, 60),
    (46, 134, 222),
    (46, 204, 113),
    (241, 196, 15),
]


def token_color(token_id: int) -> tuple[int, int, int]:
    """Couleur d'un token (wrap modulo nb de couleurs)."""
    return TOKEN_COLORS[int(token_id) % len(TOKEN_COLORS)]


def lineage_color(root_id: int) -> tuple[int, int, int]:
    """Couleur stable et déterministe d'une lignée (hash root_id → teinte).

    Espacement par nombre d'or pour maximiser la séparation des teintes.
    """
    hue = (int(root_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.92)
    return (int(r * 255), int(g * 255), int(b * 255))
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `pytest tests/test_v8_replay.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add aetherlife/viz/v8_replay.py tests/test_v8_replay.py
git commit -m "feat(viz): couleurs deterministes token/lignee (module pur)"
```

---

## Task 3: Chargement + validation du contrat

**Files:**
- Modify: `aetherlife/viz/v8_replay.py`
- Test: `tests/test_v8_replay.py`

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_v8_replay.py` :

```python
import json
from aetherlife.viz.v8_replay import (
    load_meta, iter_events, validate_event,
)


def _write(tmp_path, meta, events):
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = "\n".join(json.dumps(e) for e in events)
    (tmp_path / "events.jsonl").write_text(lines + "\n", encoding="utf-8")


def test_load_meta_roundtrip(tmp_path):
    meta = {"rows": 24, "cols": 24, "n_tokens": 4, "schema_version": 1}
    _write(tmp_path, meta, [])
    assert load_meta(str(tmp_path / "meta.json"))["rows"] == 24


def test_iter_events_counts_and_skips_blank(tmp_path):
    events = [{"t": 10, "agents": []}, {"t": 20, "agents": []}]
    _write(tmp_path, {"rows": 1, "cols": 1}, events)
    got = list(iter_events(str(tmp_path / "events.jsonl")))
    assert [e["t"] for e in got] == [10, 20]


def test_validate_event_ok():
    assert validate_event({"t": 1, "agents": []}) is True


def test_validate_event_missing_key_raises():
    import pytest
    with pytest.raises(ValueError):
        validate_event({"agents": []})
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `pytest tests/test_v8_replay.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_meta'`

- [ ] **Step 3: Implémenter le chargement**

Ajouter à la fin de `aetherlife/viz/v8_replay.py` :

```python
def load_meta(path: str) -> dict[str, Any]:
    """Charge meta.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_events(path: str) -> Iterator[dict[str, Any]]:
    """Itère les événements d'un events.jsonl (1 objet JSON par ligne non vide)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_REQUIRED_EVENT_KEYS = {"t", "agents"}


def validate_event(event: dict[str, Any]) -> bool:
    """Vérifie la présence des clés minimales du contrat."""
    missing = _REQUIRED_EVENT_KEYS - set(event.keys())
    if missing:
        raise ValueError(f"event missing keys: {sorted(missing)}")
    return True
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `pytest tests/test_v8_replay.py -v`
Expected: PASS (9 tests au total)

- [ ] **Step 5: Commit**

```bash
git add aetherlife/viz/v8_replay.py tests/test_v8_replay.py
git commit -m "feat(viz): chargement + validation du contrat events.jsonl"
```

---

## Task 4: Rendu d'une frame (`_draw_frame`)

**Files:**
- Create: `aetherlife/viz/pygame_viewer_v8.py`
- Test: `tests/test_pygame_viewer_v8.py`

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/test_pygame_viewer_v8.py` :

```python
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from aetherlife.viz.pygame_viewer_v8 import _draw_frame
from aetherlife.viz.v8_replay import lineage_color, token_color

META = {"rows": 6, "cols": 6, "n_tokens": 4}
CELL = 10


def test_frame_dimensions_include_hud():
    ev = {"t": 1, "agents": []}
    surf = _draw_frame(ev, META, cell_px=CELL)
    # largeur = cols*cell ; hauteur = rows*cell + HUD
    assert surf.get_width() == META["cols"] * CELL
    assert surf.get_height() > META["rows"] * CELL


def test_agent_cell_tinted_by_lineage():
    ev = {"t": 1, "agents": [{"id": 0, "lin": 12, "r": 2, "c": 3}]}
    surf = _draw_frame(ev, META, cell_px=CELL)
    # centre de la cellule (c=3,r=2) : x=3*10+5=35, y=2*10+5=25
    px = surf.get_at((35, 25))[:3]
    assert tuple(px) == lineage_color(12)


def test_vocalize_halo_uses_token_color():
    ev = {
        "t": 1,
        "agents": [{"id": 7, "lin": 5, "r": 2, "c": 3}],
        "vocal": {"7": 1},
    }
    surf = _draw_frame(ev, META, cell_px=CELL)
    # halo au sommet de la cellule : cx=3*10+5=35, cy=2*10+radius
    radius = max(2, CELL // 4)
    px = surf.get_at((35, 2 * CELL + radius))[:3]
    assert tuple(px) == token_color(1)
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `pytest tests/test_pygame_viewer_v8.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aetherlife.viz.pygame_viewer_v8'`

- [ ] **Step 3: Implémenter `_draw_frame`**

Créer `aetherlife/viz/pygame_viewer_v8.py` :

```python
"""Renderer V8 Replay Viewer — pur (lit le contrat, pas de torch/env).

SDL dummy par défaut : rend en offscreen, aucune fenêtre requise.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from aetherlife.viz.v8_replay import (  # noqa: E402
    iter_events, lineage_color, load_meta, token_color,
)

BG = (18, 18, 20)
SPOT = (120, 120, 130)
SPOT_HOT = (255, 230, 120)
HUD_H = 28
HUD_BG = (10, 10, 12)
HUD_FG = (210, 210, 215)


def _draw_frame(
    event: dict, meta: dict, cell_px: int = 16, focus_lineage: int | None = None
) -> "pygame.Surface":
    """Rend un tick en une Surface offscreen."""
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    rows, cols = int(meta["rows"]), int(meta["cols"])
    width, height = cols * cell_px, rows * cell_px + HUD_H
    surf = pygame.Surface((width, height))
    surf.fill(BG)

    # Gather spots (sous les agents) — surbrillance si >= 2 adjacents
    for s in event.get("spots", []):
        hot = int(s.get("n", 0)) >= 2
        rect = pygame.Rect(s["c"] * cell_px, s["r"] * cell_px, cell_px, cell_px)
        pygame.draw.rect(surf, SPOT_HOT if hot else SPOT, rect, width=2)

    # Agents teintés par lignée
    pos_by_id: dict[int, tuple[int, int]] = {}
    for a in event["agents"]:
        col = lineage_color(a["lin"])
        if focus_lineage is not None and a["lin"] != focus_lineage:
            col = (col[0] // 3, col[1] // 3, col[2] // 3)
        rect = pygame.Rect(a["c"] * cell_px, a["r"] * cell_px, cell_px, cell_px)
        pygame.draw.rect(surf, col, rect)
        pos_by_id[int(a["id"])] = (a["r"], a["c"])

    # Halos de vocalisation (couleur = token), au sommet de la cellule du speaker
    radius = max(2, cell_px // 4)
    for sid, tok in event.get("vocal", {}).items():
        rc = pos_by_id.get(int(sid))
        if rc is None:
            continue
        cx = rc[1] * cell_px + cell_px // 2
        cy = rc[0] * cell_px + radius
        pygame.draw.circle(surf, token_color(tok), (cx, cy), radius)

    # HUD
    pygame.draw.rect(surf, HUD_BG, pygame.Rect(0, rows * cell_px, width, HUD_H))
    font = pygame.font.SysFont("monospace", 14)
    txt = (
        f"t={event.get('t', 0)}  alive={event.get('n_alive', '?')}  "
        f"lin={event.get('n_lin', '?')}  season={event.get('season', '?')}"
    )
    surf.blit(font.render(txt, True, HUD_FG), (6, rows * cell_px + 6))
    return surf
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `pytest tests/test_pygame_viewer_v8.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add aetherlife/viz/pygame_viewer_v8.py tests/test_pygame_viewer_v8.py
git commit -m "feat(viz): _draw_frame — agents par lignee + halos token + spots + HUD"
```

---

## Task 5: `render_events` (frames → PNG / MP4)

**Files:**
- Modify: `aetherlife/viz/pygame_viewer_v8.py`
- Test: `tests/test_pygame_viewer_v8.py`

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_pygame_viewer_v8.py` :

```python
import json
from aetherlife.viz.pygame_viewer_v8 import render_events


def _write_run(tmp_path):
    meta = {"rows": 6, "cols": 6, "n_tokens": 4}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    events = [
        {"t": 10, "n_alive": 1, "n_lin": 1, "season": 0,
         "agents": [{"id": 0, "lin": 3, "r": 1, "c": 1}], "vocal": {}, "spots": []},
        {"t": 20, "n_alive": 1, "n_lin": 1, "season": 0,
         "agents": [{"id": 0, "lin": 3, "r": 2, "c": 2}], "vocal": {"0": 2},
         "spots": [{"r": 4, "c": 4, "n": 2}]},
        {"t": 30, "n_alive": 1, "n_lin": 1, "season": 1,
         "agents": [{"id": 0, "lin": 3, "r": 3, "c": 3}], "vocal": {}, "spots": []},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return str(tmp_path / "events.jsonl"), str(tmp_path / "meta.json")


def test_render_png_writes_one_frame_per_event(tmp_path):
    events, meta = _write_run(tmp_path)
    out_dir = str(tmp_path / "frames")
    paths = render_events(events, meta, out_dir, fmt="png", cell_px=8)
    assert len(paths) == 3
    assert all(os.path.getsize(p) > 0 for p in paths)


def test_render_png_respects_tick_range(tmp_path):
    events, meta = _write_run(tmp_path)
    out_dir = str(tmp_path / "frames2")
    paths = render_events(events, meta, out_dir, fmt="png", cell_px=8,
                          from_tick=15, to_tick=25)
    assert len(paths) == 1  # seul t=20 dans [15,25]


def test_render_mp4_produces_nonempty_file(tmp_path):
    events, meta = _write_run(tmp_path)
    out = str(tmp_path / "clip.mp4")
    res = render_events(events, meta, out, fmt="mp4", fps=5, cell_px=8)
    assert res == out
    assert os.path.getsize(out) > 0
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `pytest tests/test_pygame_viewer_v8.py::test_render_png_writes_one_frame_per_event -v`
Expected: FAIL — `ImportError: cannot import name 'render_events'`

- [ ] **Step 3: Implémenter `render_events`**

Ajouter à la fin de `aetherlife/viz/pygame_viewer_v8.py` :

```python
def render_events(
    events_path: str,
    meta_path: str,
    out_path: str,
    *,
    fmt: str = "mp4",
    fps: int = 30,
    from_tick: int = 0,
    to_tick: int | None = None,
    focus_lineage: int | None = None,
    cell_px: int = 16,
):
    """Rend un events.jsonl en frames.

    fmt='png'  → out_path est un dossier ; retourne la liste des PNG.
    fmt='gif'|'mp4' → out_path est le fichier clip ; retourne out_path.
    """
    meta = load_meta(meta_path)
    if not pygame.get_init():
        pygame.init()
    pygame.font.init()

    png_paths: list[str] = []
    video_frames = []
    if fmt == "png":
        os.makedirs(out_path, exist_ok=True)

    for ev in iter_events(events_path):
        if ev["t"] < from_tick:
            continue
        if to_tick is not None and ev["t"] > to_tick:
            break
        surf = _draw_frame(ev, meta, cell_px, focus_lineage)
        if fmt == "png":
            p = os.path.join(out_path, f"frame_{ev['t']:06d}.png")
            pygame.image.save(surf, p)
            png_paths.append(p)
        else:
            # (W,H,3) → (H,W,3) pour imageio
            video_frames.append(pygame.surfarray.array3d(surf).swapaxes(0, 1))

    if fmt == "png":
        return png_paths

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimsave(out_path, video_frames, fps=fps)
    return out_path
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `pytest tests/test_pygame_viewer_v8.py -v`
Expected: PASS (6 tests — 3 de Task 4 + 3 ici). Le test MP4 invoque ffmpeg bundlé (imageio-ffmpeg).

- [ ] **Step 5: Commit**

```bash
git add aetherlife/viz/pygame_viewer_v8.py tests/test_pygame_viewer_v8.py
git commit -m "feat(viz): render_events — PNG sequence + assemblage MP4 (imageio)"
```

---

## Task 6: CLI `render_v8.py`

**Files:**
- Create: `scripts/render_v8.py`
- Test: `tests/test_pygame_viewer_v8.py`

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_pygame_viewer_v8.py` :

```python
import subprocess
import sys


def test_render_v8_cli_end_to_end(tmp_path):
    events, meta = _write_run(tmp_path)
    out = str(tmp_path / "cli_clip.mp4")
    r = subprocess.run(
        [sys.executable, "scripts/render_v8.py", "--events", events,
         "--out", out, "--fps", "5", "--cell-px", "8"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert os.path.getsize(out) > 0
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `pytest tests/test_pygame_viewer_v8.py::test_render_v8_cli_end_to_end -v`
Expected: FAIL — returncode != 0 (`scripts/render_v8.py` n'existe pas)

- [ ] **Step 3: Implémenter la CLI**

Créer `scripts/render_v8.py` :

```python
"""CLI — rend un events.jsonl V8 en clip PNG/GIF/MP4.

Usage:
    python scripts/render_v8.py --events results/seed25/events.jsonl \\
        --out clips/seed25.mp4 --fps 30 [--from-tick 0 --to-tick 16000] \\
        [--focus-lineage 12] [--cell-px 16]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aetherlife.viz.pygame_viewer_v8 import render_events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--meta", default=None, help="défaut: meta.json à côté de --events")
    p.add_argument("--out", required=True, help="fichier .mp4/.gif, ou dossier si .png")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--from-tick", type=int, default=0)
    p.add_argument("--to-tick", type=int, default=None)
    p.add_argument("--focus-lineage", type=int, default=None)
    p.add_argument("--cell-px", type=int, default=16)
    a = p.parse_args()

    meta = a.meta or os.path.join(os.path.dirname(a.events), "meta.json")
    ext = os.path.splitext(a.out)[1].lower().lstrip(".")
    fmt = ext if ext in ("png", "gif", "mp4") else "mp4"
    res = render_events(
        a.events, meta, a.out, fmt=fmt, fps=a.fps,
        from_tick=a.from_tick, to_tick=a.to_tick,
        focus_lineage=a.focus_lineage, cell_px=a.cell_px,
    )
    print(f"WROTE {res}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `pytest tests/test_pygame_viewer_v8.py::test_render_v8_cli_end_to_end -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/render_v8.py tests/test_pygame_viewer_v8.py
git commit -m "feat(viz): CLI render_v8 (events.jsonl -> clip mp4/gif/png)"
```

---

## Task 7: Recorder `record_events_v8.py` (GPU)

**Files:**
- Create: `scripts/record_events_v8.py`
- Test: `tests/test_record_events_v8.py`

> Le recorder **réutilise** `build_env` et reproduit **verbatim** le setup
> `BrainConfig` + `vision_radius` de `overnight_v8b1.run` (lignes 392-402) pour
> que la trajectoire enregistrée corresponde au régime réellement étudié.
> **Ne modifie pas** `overnight_v8b1.py`. ⚠️ `coordination_collective` tourne en
> `vision_radius=4` (la liste ligne 392 ne l'inclut pas) — copie fidèle.

- [ ] **Step 1: Écrire le smoke test qui échoue**

Créer `tests/test_record_events_v8.py` :

```python
"""Smoke recorder — court, CPU, vérifie le contrat produit."""
import json
import os
import subprocess
import sys


def test_recorder_produces_valid_contract(tmp_path):
    out_dir = str(tmp_path / "rec")
    r = subprocess.run(
        [sys.executable, "scripts/record_events_v8.py",
         "--seed", "1", "--ticks", "40", "--record-every", "10",
         "--regime", "coordination_collective", "--device", "cpu",
         "--out-dir", out_dir],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr
    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    assert meta["total_ticks"] == 40 and meta["record_every"] == 10
    assert meta["schema_version"] == 1
    lines = [
        json.loads(x)
        for x in open(os.path.join(out_dir, "events.jsonl"), encoding="utf-8")
        if x.strip()
    ]
    assert len(lines) >= 1
    ev = lines[0]
    assert {"t", "agents", "vocal", "spots", "n_alive"} <= set(ev.keys())
    if ev["agents"]:
        a = ev["agents"][0]
        assert {"id", "lin", "r", "c", "e", "er", "age", "aff"} <= set(a.keys())
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `pytest tests/test_record_events_v8.py -v`
Expected: FAIL — returncode != 0 (`scripts/record_events_v8.py` absent)

- [ ] **Step 3: Implémenter le recorder**

Créer `scripts/record_events_v8.py` :

```python
"""Recorder — re-simule un seed V8-C3 et dumpe events.jsonl + meta.json.

Réutilise build_env de overnight_v8b1 SANS le modifier. Reproduit verbatim le
setup BrainConfig + vision_radius de overnight_v8b1.run.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/record_events_v8.py \\
        --seed 25 --ticks 16000 --record-every 10 \\
        --regime coordination_collective --device cuda --out-dir results/clip_seed25
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overnight_v8b1 import build_env  # noqa: E402
from aetherlife.agents.lineage_agent import LineageAgent  # noqa: E402
from aetherlife.agents.lineage_brain import BrainConfig  # noqa: E402


def _spot_adjacency(env, pos) -> int:
    """Nb d'agents vivants à distance Manhattan <= 1 du spot."""
    n = 0
    for a in env._agents:  # noqa: SLF001
        if a.alive and abs(a.pos[0] - pos[0]) + abs(a.pos[1] - pos[1]) <= 1:
            n += 1
    return n


def record(
    seed: int,
    *,
    regime: str = "coordination_collective",
    ticks: int = 16000,
    vocalize_cost: float = 0.05,
    max_pop_override: int | None = None,
    bonus_energy_override: float | None = None,
    record_every: int = 10,
    out_dir: str = "results/clip",
    device: str = "cuda",
) -> str:
    env = build_env(
        seed, regime=regime, vocalize_energy_cost=vocalize_cost,
        max_pop_override=max_pop_override,
        bonus_energy_override=bonus_energy_override,
    )
    # Mirror verbatim de overnight_v8b1.run (lignes 392-402)
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

    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "rows": env.cfg.rows, "cols": env.cfg.cols,
        "n_tokens": env.cfg.vocabulary.n_tokens,
        "listen_radius": env.cfg.vocabulary.listen_radius,
        "seed": seed, "regime": regime, "vcost": vocalize_cost,
        "total_ticks": ticks, "record_every": record_every,
        "schema_version": 1,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    path = os.path.join(out_dir, "events.jsonl")
    with open(path, "w", encoding="utf-8") as out:
        for t in range(1, ticks + 1):
            if env.n_alive == 0:
                break
            obs_stub = {
                a.agent_id: np.zeros(10)
                for a in env._agents if a.alive  # noqa: SLF001
            }
            actions = policy.act_dict(obs_stub, greedy=False)
            env.step(actions)
            if t % record_every != 0:
                continue
            agents = []
            for a in env._agents:  # noqa: SLF001
                if not a.alive:
                    continue
                agents.append({
                    "id": a.agent_id, "lin": a.root_ancestor_id,
                    "r": a.pos[0], "c": a.pos[1],
                    "e": round(float(a.energy), 1),
                    "er": round(float(a.energy) / float(env.cfg.max_energy), 3),
                    "age": t - a.birth_tick,
                    "aff": a.biome_affinity,
                })
            vocal = {
                str(sid): int(tid)
                for sid, tid in getattr(env, "_tokens_this_tick", {}).items()
            }
            spots = [
                {"r": pos[0], "c": pos[1], "n": _spot_adjacency(env, pos)}
                for pos in env.gather_spots
            ]
            ev = {
                "t": t, "season": int(env.season), "n_alive": env.n_alive,
                "n_lin": len(policy.registry),
                "agents": agents, "vocal": vocal, "spots": spots,
            }
            out.write(json.dumps(ev) + "\n")
    print(f"WROTE {path}  ({out_dir}/meta.json)")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--regime", default="coordination_collective")
    p.add_argument("--ticks", type=int, default=16000)
    p.add_argument("--vocalize-cost", type=float, default=0.05)
    p.add_argument("--max-pop-override", type=int, default=None)
    p.add_argument("--bonus-energy-override", type=float, default=None)
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--out-dir", default="results/clip")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    record(
        a.seed, regime=a.regime, ticks=a.ticks, vocalize_cost=a.vocalize_cost,
        max_pop_override=a.max_pop_override,
        bonus_energy_override=a.bonus_energy_override,
        record_every=a.record_every, out_dir=a.out_dir, device=a.device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le smoke test pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_record_events_v8.py -v`
Expected: PASS (le run CPU 40 ticks reste court). Si lent, c'est attendu (brains lineage) ; timeout fixé à 600 s.

- [ ] **Step 5: Commit**

```bash
git add scripts/record_events_v8.py tests/test_record_events_v8.py
git commit -m "feat(viz): recorder events.jsonl (reutilise build_env, runner intact)"
```

---

## Task 8: Vérification finale + clip de démonstration

**Files:** aucun nouveau fichier (validation end-to-end).

- [ ] **Step 1: Suite complète viz verte**

Run: `pytest tests/test_v8_replay.py tests/test_pygame_viewer_v8.py tests/test_record_events_v8.py -v`
Expected: tous PASS.

- [ ] **Step 2: Non-régression globale**

Run: `pytest -q`
Expected: les 453 tests existants + les nouveaux passent (aucune régression — le viewer n'touche pas au core).

- [ ] **Step 3: Générer le clip de démo seed25 (GPU)**

Run:
```bash
PYTHONIOENCODING=utf-8 python scripts/record_events_v8.py --seed 25 \
    --regime coordination_collective --ticks 16000 --record-every 10 \
    --device cuda --out-dir results/clip_seed25
python scripts/render_v8.py --events results/clip_seed25/events.jsonl \
    --out clips/seed25.mp4 --fps 30 --cell-px 16
```
Expected: `WROTE clips/seed25.mp4`. Visionner : on voit les lignées colorées, les halos de tokens, les gather spots en surbrillance.

- [ ] **Step 4: Documenter le clip**

Ajouter une ligne dans le finding coordination (ou un court `clips/README.md`) pointant vers le clip seed25 et la commande de régénération. Commit.

```bash
git add clips/README.md
git commit -m "docs(viz): clip de demo seed25 + commande de regeneration"
```

---

## Self-Review (rempli par l'auteur du plan)

- **Couverture spec** : contrat events.jsonl (T2-3) ✓ · recorder réutilise build_env sans modif (T7) ✓ · renderer pur testable sans GPU (T4-5) ✓ · overlays lignée/token/spot/HUD (T4) ✓ · CLI (T6) ✓ · extra viz (T1) ✓ · seeds démo seed25 (T8) ✓ · champs er/age/aff (T7) ✓ · record_every=10 défaut (T7) ✓.
- **Pas de placeholder** : tout le code est complet et exécutable.
- **Cohérence des types** : `render_events(events_path, meta_path, out_path, *, fmt, ...)` identique en T5/T6 ; `_draw_frame(event, meta, cell_px, focus_lineage)` identique T4/T5 ; clés agent `{id,lin,r,c,e,er,age,aff}` identiques recorder (T7) et tests renderer (T4).
- **Écart assumé vs spec** : `n_births` retiré du schéma par-tick (non disponible sans modifier le core ; le HUD tolère l'absence). À acter si tu veux le champ → nécessiterait un compteur dans le recorder.
