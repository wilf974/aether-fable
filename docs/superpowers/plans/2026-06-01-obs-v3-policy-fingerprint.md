# OBS Viewer 3.0 — Policy Fingerprint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mesurer et comparer les politiques apprises des lignées (Q-values sur une batterie de sondes) pour tester H2 (village vs mobile = politiques différentes ?) via une Policy Distance et des heatmaps comparatives.

**Architecture:** Moteur pur (sondes construites via le VRAI `egocentric_obs` sur un env numpy contrôlé → layer-safe ; fingerprint = forward du brain ; distance cosine). Script de capture in-process (run un seed, auto-étiquette sa mobilité, sonde le cerveau dominant). Script de rendu (Policy Distance + heatmaps pygame).

**Tech Stack:** Python 3.13, numpy, torch (brain forward), pygame (heatmaps), pytest. Le moteur + viz testables CPU sans GPU.

**Spec :** `docs/superpowers/specs/2026-06-01-obs-viewer3-policy-fingerprint-design.md`

**Faits API (vérifiés)** :
- `egocentric_obs(env, agent, vision_radius, *, listener_vocab=None, embedding_dim=0)` → np.ndarray. Canaux spatiaux (food/nests/plants/agents/biome/gather) en fenêtre 9×9 (r=4) + [energy_norm, age_norm, season_phase] + heard_embeddings (16) → 505 dims en coordination.
- heard tokens lus de `env._tokens_this_tick` (dict agent_id→token), voisins ≤ listen_radius, décodés via `listener_vocab.get_embedding(tok)`.
- `LineageBrain(root_id, obs_dim, n_actions, cfg, *, seed=, vocabulary=)`. `brain.online(tensor)` → Q-values (9). `brain._torch` = torch.
- `registry.get(root_id)` / `iter(registry)`. Lignée d'un agent = `agent.root_ancestor_id`.
- env pur numpy : `env._food_mask` (bool array), `env._agents` (list `_AgentState`), `env._gather_spots`, `env._tokens_this_tick`, `env.step_count`, `env.phase`, `env.cfg.{rows,cols,max_energy,max_steps,vision...}`.
- `_AgentState` (import `aetherlife.world.multi_agent_grid`) : `agent_id, pos, energy, alive, root_ancestor_id, birth_tick, biome_affinity`.
- coordination_collective : vision_radius=4, vocab embedding_dim=16, 9 actions `[MOVE0-3, VOC0-3, GATHER]`.

---

## File Structure

| Fichier | Responsabilité |
|---|---|
| `aetherlife/viz/policy_probe.py` | batterie de sondes, `build_probe_obs`, `fingerprint`, `policy_distance`, labels |
| `scripts/probe_policies_v8.py` | capture : run seed + mobility_score + sonde cerveau dominant → JSON |
| `scripts/render_policy_compare.py` | Policy Distance + heatmaps comparatives PNG |
| `tests/test_policy_probe.py` | moteur + distance + viz (CPU) |
| `tests/test_probe_policies_v8.py` | smoke capture |

---

## Task 1: Batterie de sondes + `build_probe_obs` (layer-safe)

**Files:**
- Create: `aetherlife/viz/policy_probe.py`
- Test: `tests/test_policy_probe.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_policy_probe.py` :

```python
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import numpy as np

from aetherlife.viz.policy_probe import (
    ACTION_LABELS, PROBE_LABELS, build_probe_obs, make_probe_env,
)


def test_action_labels_are_9():
    assert len(ACTION_LABELS) == 9
    assert ACTION_LABELS[0].startswith("MOVE")
    assert ACTION_LABELS[8] == "GATHER"


def test_probe_labels_include_core_scenarios():
    for lab in ("Food_N", "Food_S", "Food_E", "Food_W", "Alone",
                "Dense_neighbors", "Low_energy", "High_energy",
                "Gather_adjacent", "Token_heard_0", "Token_heard_1"):
        assert lab in PROBE_LABELS


def test_build_probe_obs_dim_is_505():
    env = make_probe_env(seed=1)
    obs = build_probe_obs(env, "Food_N")
    assert obs.shape == (505,)
    assert obs.dtype == np.float32


def test_food_north_activates_food_channel_north_cell():
    # food_view est le 1er canal (81 valeurs, fenêtre 9x9, r=4, centre index 40).
    # Food_N place une food au nord -> cellule (centre_r - k, centre_c) du canal.
    env = make_probe_env(seed=1)
    obs = build_probe_obs(env, "Food_N")
    food_channel = obs[:81].reshape(9, 9)
    # au moins une cellule de la colonne centrale, au nord du centre, est active
    north_col = food_channel[:4, 4]
    assert north_col.sum() >= 1.0
    # et la moitié sud de la colonne centrale est vide
    assert food_channel[5:, 4].sum() == 0.0


def test_alone_has_no_agents_channel():
    env = make_probe_env(seed=1)
    obs = build_probe_obs(env, "Alone")
    agents_channel = obs[3 * 81:4 * 81]  # 4e canal = autres agents
    assert agents_channel.sum() == 0.0


def test_dense_has_agents_channel():
    env = make_probe_env(seed=1)
    obs = build_probe_obs(env, "Dense_neighbors")
    agents_channel = obs[3 * 81:4 * 81]
    assert agents_channel.sum() >= 1.0


def test_low_vs_high_energy_differ():
    env = make_probe_env(seed=1)
    lo = build_probe_obs(env, "Low_energy")
    hi = build_probe_obs(env, "High_energy")
    # energy_norm est le 1er des 3 scalaires, après les canaux spatiaux
    n_spatial = 6 * 81  # 6 canaux (coop actif)
    assert lo[n_spatial] < hi[n_spatial]
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_policy_probe.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aetherlife.viz.policy_probe'`

- [ ] **Step 3: Implémenter le moteur de sondes**

Créer `aetherlife/viz/policy_probe.py` :

```python
"""OBS V3.0 — Policy Fingerprint : sondes synthétiques + empreinte + distance.

Teste H2 (village vs mobile = politiques apprises différentes ?). Les sondes sont
construites via le VRAI `egocentric_obs` sur un env numpy contrôlé → garantit que
le vecteur 505-dim correspond exactement à ce sur quoi le cerveau s'est entraîné.
"""
from __future__ import annotations

import numpy as np

from aetherlife.agents.lineage_agent import egocentric_obs
from aetherlife.world.multi_agent_grid import _AgentState

# Espace d'actions coordination_collective : 4 moves + 4 vocalize + 1 gather.
ACTION_LABELS = [
    "MOVE_0", "MOVE_1", "MOVE_2", "MOVE_3",
    "VOC_0", "VOC_1", "VOC_2", "VOC_3", "GATHER",
]

PROBE_LABELS = [
    "Food_N", "Food_S", "Food_E", "Food_W",
    "Gather_adjacent", "Token_heard_0", "Token_heard_1",
    "Low_energy", "High_energy", "Alone", "Dense_neighbors",
]

_VISION = 4
_EMB = 16


def make_probe_env(seed: int = 1):
    """Construit un env coordination_collective (numpy, CPU) pour les sondes."""
    import sys, os
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "scripts"),
    )
    from overnight_v8b1 import build_env
    env = build_env(seed, regime="coordination_collective",
                    vocalize_energy_cost=0.05)
    env.reset(seed=seed)
    return env


def _clean_center_agent(env) -> _AgentState:
    """Réinitialise l'env à un état contrôlé : 1 agent au centre, pas de food,
    pas de voisins, pas de tokens, pas de spots. Retourne l'agent-sonde."""
    rows, cols = env.cfg.rows, env.cfg.cols
    cr, cc = rows // 2, cols // 2
    env._food_mask[:] = False  # noqa: SLF001
    env._tokens_this_tick = {}  # noqa: SLF001
    env._gather_spots = {}  # noqa: SLF001
    agent = _AgentState(
        agent_id=0, pos=(cr, cc), energy=env.cfg.max_energy * 0.5,
        alive=True, root_ancestor_id=0, birth_tick=0,
        biome_affinity=0,
    )
    env._agents = [agent]  # noqa: SLF001
    return agent


def _add_neighbor(env, dr: int, dc: int, agent_id: int = 1) -> _AgentState:
    ar, ac = env._agents[0].pos  # noqa: SLF001
    nb = _AgentState(
        agent_id=agent_id, pos=(ar + dr, ac + dc),
        energy=env.cfg.max_energy * 0.5, alive=True,
        root_ancestor_id=1, birth_tick=0, biome_affinity=0,
    )
    env._agents.append(nb)  # noqa: SLF001
    return nb


def build_probe_obs(env, label: str, *, listener_vocab=None) -> np.ndarray:
    """Construit l'observation d'une sonde via le vrai egocentric_obs."""
    agent = _clean_center_agent(env)
    ar, ac = agent.pos
    if label == "Food_N":
        env._food_mask[ar - 2, ac] = True  # noqa: SLF001
    elif label == "Food_S":
        env._food_mask[ar + 2, ac] = True  # noqa: SLF001
    elif label == "Food_E":
        env._food_mask[ar, ac + 2] = True  # noqa: SLF001
    elif label == "Food_W":
        env._food_mask[ar, ac - 2] = True  # noqa: SLF001
    elif label == "Gather_adjacent":
        from aetherlife.world.cooperative import GatherSpot
        spot_pos = (ar, ac + 1)
        env._gather_spots = {  # noqa: SLF001
            spot_pos: GatherSpot(pos=spot_pos, ticks_left=50),
        }
        _add_neighbor(env, 0, 1, agent_id=1)
    elif label == "Token_heard_0":
        nb = _add_neighbor(env, 0, 1, agent_id=1)
        env._tokens_this_tick = {nb.agent_id: 0}  # noqa: SLF001
    elif label == "Token_heard_1":
        nb = _add_neighbor(env, 0, 1, agent_id=1)
        env._tokens_this_tick = {nb.agent_id: 1}  # noqa: SLF001
    elif label == "Low_energy":
        agent.energy = env.cfg.max_energy * 0.1
    elif label == "High_energy":
        agent.energy = env.cfg.max_energy * 0.9
    elif label == "Alone":
        pass  # déjà seul
    elif label == "Dense_neighbors":
        for i, (dr, dc) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
            _add_neighbor(env, dr, dc, agent_id=i + 1)
    else:
        raise ValueError(f"sonde inconnue : {label}")
    obs = egocentric_obs(
        env, agent, _VISION, listener_vocab=listener_vocab, embedding_dim=_EMB,
    )
    return obs.astype(np.float32)
```

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_policy_probe.py -q`
Expected: PASS (7 tests). `make_probe_env` construit un env numpy (CPU, pas de GPU).

- [ ] **Step 5: Commit**

```bash
git add aetherlife/viz/policy_probe.py tests/test_policy_probe.py
git commit -m "feat(obs-v3): batterie de sondes + build_probe_obs (layer-safe via egocentric_obs)"
```

---

## Task 2: `fingerprint` + `policy_distance`

**Files:**
- Modify: `aetherlife/viz/policy_probe.py`
- Test: `tests/test_policy_probe.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_policy_probe.py` :

```python
from aetherlife.viz.policy_probe import fingerprint, policy_distance


def _make_cpu_brain(obs_dim=505, n_actions=9):
    from aetherlife.agents.lineage_brain import BrainConfig, LineageBrain
    cfg = BrainConfig(enabled=True, device="cpu", vision_radius=4,
                      hidden_dims=(64, 64))
    return LineageBrain(root_id=0, obs_dim=obs_dim, n_actions=n_actions, cfg=cfg,
                        seed=0)


def test_fingerprint_shape():
    env = make_probe_env(seed=1)
    brain = _make_cpu_brain()
    fp = fingerprint(brain, env)
    assert fp.shape == (len(PROBE_LABELS), 9)
    assert np.isfinite(fp).all()


def test_policy_distance_identical_is_zero():
    fp = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    assert policy_distance(fp, fp) == 0.0


def test_policy_distance_symmetric():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert policy_distance(a, b) == policy_distance(b, a)


def test_policy_distance_orthogonal_is_one():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert abs(policy_distance(a, b) - 1.0) < 1e-9
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_policy_probe.py::test_policy_distance_orthogonal_is_one -q`
Expected: FAIL — `ImportError: cannot import name 'fingerprint'`

- [ ] **Step 3: Implémenter fingerprint + distance**

Ajouter à la fin de `aetherlife/viz/policy_probe.py` :

```python
def fingerprint(brain, env) -> np.ndarray:
    """Matrice (n_sondes × 9) des Q-values du brain sur la batterie de sondes."""
    torch = brain._torch  # noqa: SLF001
    rows = []
    for label in PROBE_LABELS:
        obs = build_probe_obs(env, label, listener_vocab=brain.vocabulary)
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(brain.device)
            q = brain.online(x).cpu().numpy().reshape(-1)
        rows.append(q)
    return np.array(rows, dtype=np.float32)


def policy_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Distance cosine entre deux empreintes aplaties. 0=identique, 1=orthogonal."""
    a = np.asarray(fp_a, dtype=np.float64).reshape(-1)
    b = np.asarray(fp_b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos
```

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_policy_probe.py -q`
Expected: PASS (11 tests). `test_fingerprint_shape` construit un brain CPU et le sonde (pas de GPU).

- [ ] **Step 5: Commit**

```bash
git add aetherlife/viz/policy_probe.py tests/test_policy_probe.py
git commit -m "feat(obs-v3): fingerprint (Q-values sur sondes) + policy_distance (cosine)"
```

---

## Task 3: `probe_policies_v8.py` (capture)

**Files:**
- Create: `scripts/probe_policies_v8.py`
- Test: `tests/test_probe_policies_v8.py`

> Réutilise `build_env` + `LineageAgent` (runner taggé intact). Run un seed,
> accumule l'occupation (`spatial_mobility.window_bounds`) → `mobility_score`,
> identifie la lignée dominante survivante, sonde son cerveau → JSON.

- [ ] **Step 1: Écrire le smoke test qui échoue**

Créer `tests/test_probe_policies_v8.py` :

```python
import json
import os
import subprocess
import sys


def test_probe_capture_produces_valid_json(tmp_path):
    out = str(tmp_path / "probe_seed1.json")
    r = subprocess.run(
        [sys.executable, "scripts/probe_policies_v8.py",
         "--seed", "1", "--ticks", "60", "--device", "cpu", "--out", out],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr
    with open(out, encoding="utf-8") as f:
        d = json.load(f)
    assert d["seed"] == 1
    assert "mobility_score" in d and "village_basin" in d
    assert len(d["action_labels"]) == 9
    assert len(d["fingerprint"]) == len(d["probe_labels"])
    assert all(len(row) == 9 for row in d["fingerprint"])
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_probe_policies_v8.py -q`
Expected: FAIL — returncode != 0 (`scripts/probe_policies_v8.py` absent)

- [ ] **Step 3: Implémenter la capture**

Créer `scripts/probe_policies_v8.py` :

```python
"""OBS V3.0 — capture : run un seed, calcule mobility_score, sonde le cerveau
de la lignée dominante survivante → JSON. Réutilise build_env (runner intact).

Usage:
    python scripts/probe_policies_v8.py --seed 25 --ticks 16000 \\
        --device cuda --out results/probe/seed25.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overnight_v8b1 import build_env  # noqa: E402
from aetherlife.agents.lineage_agent import LineageAgent  # noqa: E402
from aetherlife.agents.lineage_brain import BrainConfig  # noqa: E402
from aetherlife.historian.spatial_mobility import (  # noqa: E402
    OccupancyAccumulator, build_spatial_mobility_block, window_bounds,
)
from aetherlife.viz.policy_probe import (  # noqa: E402
    ACTION_LABELS, PROBE_LABELS, fingerprint, make_probe_env,
)


def probe(seed: int, *, ticks: int = 16000,
          regime: str = "coordination_collective",
          device: str = "cuda", out: str = "results/probe/seed.json") -> str:
    env = build_env(seed, regime=regime, vocalize_energy_cost=0.05)
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

    swin, ewin = window_bounds(ticks)
    occ_s = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    occ_e = OccupancyAccumulator(env.cfg.rows, env.cfg.cols)
    for t in range(1, ticks + 1):
        if env.n_alive == 0:
            break
        obs_stub = {a.agent_id: np.zeros(10)
                    for a in env._agents if a.alive}  # noqa: SLF001
        actions = policy.act_dict(obs_stub, greedy=False)
        env.step(actions)
        if swin[0] < t <= swin[1] or ewin[0] < t <= ewin[1]:
            pos = [(a.pos[0], a.pos[1])
                   for a in env._agents if a.alive]  # noqa: SLF001
            (occ_s if t <= swin[1] else occ_e).add_positions(pos)

    sm = build_spatial_mobility_block(occ_s, occ_e,
                                      start_window=swin, end_window=ewin)

    # lignée dominante survivante
    alive_roots = Counter(
        a.root_ancestor_id for a in env._agents if a.alive  # noqa: SLF001
    )
    if not alive_roots:
        raise RuntimeError(f"seed {seed} : extinction, pas de cerveau à sonder")
    dom_root = alive_roots.most_common(1)[0][0]
    brain = policy.registry.get(dom_root)
    if brain is None:
        raise RuntimeError(f"seed {seed} : brain dominant {dom_root} introuvable")

    penv = make_probe_env(seed=seed)
    fp = fingerprint(brain, penv)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    record = {
        "seed": seed, "ticks": ticks, "regime": regime,
        "mobility_score": sm["corr_occupation_start_end"],
        "village_basin": sm["village_basin"],
        "dominant_lineage": int(dom_root),
        "n_alive_final": int(env.n_alive),
        "action_labels": ACTION_LABELS,
        "probe_labels": PROBE_LABELS,
        "fingerprint": fp.tolist(),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f)
    print(f"WROTE {out}  mobility={record['mobility_score']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ticks", type=int, default=16000)
    p.add_argument("--regime", default="coordination_collective")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/probe/seed.json")
    a = p.parse_args()
    probe(a.seed, ticks=a.ticks, regime=a.regime, device=a.device, out=a.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le smoke pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_probe_policies_v8.py -q`
Expected: PASS (run CPU 60 ticks, quelques secondes ; si extinction à 60 ticks improbable mais possible, le test relancerait — seed 1 survit typiquement à 60 ticks).

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_policies_v8.py tests/test_probe_policies_v8.py
git commit -m "feat(obs-v3): capture probe_policies_v8 (run + mobility + cerveau dominant)"
```

---

## Task 4: `render_policy_compare.py` (Policy Distance + heatmaps)

**Files:**
- Create: `scripts/render_policy_compare.py`
- Test: `tests/test_policy_probe.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_policy_probe.py` :

```python
import json


def _write_probe_json(path, seed, mobility, village, fp):
    rec = {
        "seed": seed, "mobility_score": mobility, "village_basin": village,
        "action_labels": ACTION_LABELS, "probe_labels": PROBE_LABELS,
        "fingerprint": fp,
    }
    path.write_text(json.dumps(rec), encoding="utf-8")


def test_render_compare_distance_and_png(tmp_path):
    sys.path.insert(0, "scripts")
    from render_policy_compare import compare

    n_p = len(PROBE_LABELS)
    vil = [[1.0] * 9 for _ in range(n_p)]   # village ~ uniforme haut
    mob = [[0.0, 5.0] + [0.0] * 7 for _ in range(n_p)]  # mobile ~ action 1
    for i in range(3):
        _write_probe_json(tmp_path / f"v{i}.json", i, 0.95, True, vil)
    for i in range(3):
        _write_probe_json(tmp_path / f"m{i}.json", 10 + i, 0.20, False, mob)
    out_png = str(tmp_path / "cmp.png")
    res = compare([str(p) for p in tmp_path.glob("*.json")], out_png)
    assert os.path.getsize(out_png) > 0
    # village/mobile très différents -> inter >> intra
    assert res["inter"] > res["intra"]
    assert res["n_village"] == 3 and res["n_mobile"] == 3
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_policy_probe.py::test_render_compare_distance_and_png -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'render_policy_compare'`

- [ ] **Step 3: Implémenter le rendu**

Créer `scripts/render_policy_compare.py` :

```python
"""OBS V3.0 — Policy Distance + heatmaps comparatives village vs mobile.

Charge N JSON (probe_policies_v8), sépare village/mobile, calcule la distance
intra-groupe vs inter-groupe (test H2 vs H3), rend des heatmaps PNG.

Usage:
    python scripts/render_policy_compare.py results/probe/*.json --out clips/policy_compare.png
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aetherlife.viz.policy_probe import policy_distance  # noqa: E402

_CELL = 28
_PAD = 4


def _heat_color(v: float, lo: float, hi: float):
    t = 0.0 if hi == lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    return (int(30 + 200 * t), int(30 + 120 * t), int(80 + 60 * (1 - t)))


def _draw_fp(surf, fp, x0, y0, lo, hi):
    for i, row in enumerate(fp):
        for j, v in enumerate(row):
            rect = pygame.Rect(x0 + j * _CELL, y0 + i * _CELL,
                               _CELL - 1, _CELL - 1)
            pygame.draw.rect(surf, _heat_color(v, lo, hi), rect)


def compare(paths: list[str], out_png: str) -> dict:
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs.append(json.load(f))
    vil = [np.array(r["fingerprint"]) for r in recs if r["village_basin"]]
    mob = [np.array(r["fingerprint"]) for r in recs if not r["village_basin"]]

    def _pairs(group):
        return [policy_distance(group[i], group[j])
                for i in range(len(group)) for j in range(i + 1, len(group))]
    intra = _pairs(vil) + _pairs(mob)
    inter = [policy_distance(a, b) for a in vil for b in mob]
    intra_m = st.mean(intra) if intra else 0.0
    inter_m = st.mean(inter) if inter else 0.0

    # heatmaps : empreinte moyenne village vs mobile, côte à côte
    pygame.init()
    n_p = len(recs[0]["probe_labels"]) if recs else 0
    w = 2 * (9 * _CELL) + 3 * _PAD + 40
    h = n_p * _CELL + 60
    surf = pygame.Surface((w, h))
    surf.fill((18, 18, 20))
    if vil:
        fv = np.mean(vil, axis=0)
        lo, hi = float(fv.min()), float(fv.max())
        _draw_fp(surf, fv, _PAD, 40, lo, hi)
    if mob:
        fm = np.mean(mob, axis=0)
        lo, hi = float(fm.min()), float(fm.max())
        _draw_fp(surf, fm, 9 * _CELL + 2 * _PAD, 40, lo, hi)
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 14)
    surf.blit(font.render(
        f"VILLAGE (n={len(vil)})   |   MOBILE (n={len(mob)})", True,
        (220, 220, 225)), (_PAD, 6))
    surf.blit(font.render(
        f"intra={intra_m:.3f}  inter={inter_m:.3f}  "
        f"verdict={'H2' if inter_m > 1.3 * intra_m else 'H3'}",
        True, (220, 220, 225)), (_PAD, 22))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    pygame.image.save(surf, out_png)

    return {
        "n_village": len(vil), "n_mobile": len(mob),
        "intra": round(intra_m, 4), "inter": round(inter_m, 4),
        "verdict": "H2" if inter_m > 1.3 * intra_m else "H3",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="clips/policy_compare.png")
    a = p.parse_args()
    res = compare(a.paths, a.out)
    print(f"village={res['n_village']} mobile={res['n_mobile']}")
    print(f"intra-group distance = {res['intra']}")
    print(f"inter-group distance = {res['inter']}")
    print(f"VERDICT : {res['verdict']}  (inter >> intra -> H2 ; "
          f"inter ~ intra -> H3)")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer pour vérifier le succès**

Run: `PYTHONIOENCODING=utf-8 pytest tests/test_policy_probe.py -q`
Expected: PASS (tous : moteur + distance + render).

- [ ] **Step 5: Commit**

```bash
git add scripts/render_policy_compare.py tests/test_policy_probe.py
git commit -m "feat(obs-v3): render_policy_compare — Policy Distance + heatmaps village/mobile"
```

---

## Task 5: Vérification finale + batch + finding

**Files:** aucun nouveau (validation + run GPU).

- [ ] **Step 1: Suite complète verte**

Run: `PYTHONIOENCODING=utf-8 pytest -q`
Expected: tous verts (existants + nouveaux OBS V3), zéro régression.

- [ ] **Step 2: Batch capture (GPU, ~1h, arrière-plan)**

Run (bash, arrière-plan) :
```bash
for s in 25 14 24 40 31 42 1 3 6 13 16 19 20 23 27 29 32 37 46 47; do
  PYTHONIOENCODING=utf-8 python scripts/probe_policies_v8.py --seed $s \
    --ticks 16000 --device cuda --out results/probe/seed$s.json
done
```
20 runs (auto-étiquetage mobilité + empreinte cerveau dominant).

- [ ] **Step 3: Policy Distance + heatmaps**

Run:
```bash
python scripts/render_policy_compare.py results/probe/seed*.json --out clips/policy_compare.png
```
Lire : `inter >> intra` → H2 (politiques village/mobile divergent) ; `inter ~ intra`
→ H3 (contingence). Visionner `clips/policy_compare.png`.

- [ ] **Step 4: Écrire le finding H2**

Selon le résultat, écrire `docs/findings/2026-06-XX-finding-v8c3-h2-policy-divergence.md`
(H2 confirmé : politiques divergent / H3 : contingence) + mettre à jour finding
mobilité §7 + SYNTHESIS. Commit.

---

## Self-Review (auteur du plan)

- **Couverture spec** : moteur sondes + build_probe_obs layer-safe (T1) ✓ · fingerprint + policy_distance (T2) ✓ · capture in-process + auto-étiquetage mobilité + lignée dominante (T3) ✓ · render distance + heatmaps (T4) ✓ · batch 20 seeds + finding (T5) ✓ · 9 actions / 11 sondes ✓.
- **Écarts assumés** : seuil verdict `inter > 1.3×intra` = heuristique (le finding lira les chiffres bruts, pas un booléen). `make_probe_env` re-build un env par seed (léger, CPU). `step_count`/`phase` non fixés dans build_probe_obs → valeurs du reset (constantes entre sondes d'un même env → n'affectent pas les différences inter-sondes ni la distance).
- **Pas de placeholder** : code complet.
- **Cohérence types** : `fingerprint(brain, env)->np.ndarray (n_p,9)` ; `policy_distance(fp,fp)->float` ; `build_probe_obs(env,label,*,listener_vocab)->np.ndarray(505,)` ; clés JSON identiques entre capture (T3) et render (T4) : `village_basin, fingerprint, probe_labels, action_labels, mobility_score`.
