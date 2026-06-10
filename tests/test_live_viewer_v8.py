import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

pytest.importorskip("pygame", reason="suite complete : requiert pygame")


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


def test_launch_gui_v8_smoke():
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
