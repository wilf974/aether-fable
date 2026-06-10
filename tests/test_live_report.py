import os
import sys
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

pytest.importorskip("torch", reason="suite complete : requiert torch")


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
    assert isinstance(h.discoveries, list)  # pas de crash, liste (>=0)
