import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import numpy as np
import pytest

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


def test_probe_obs_identical_across_seeds_after_biome_neutralized():
    # Pour que policy_distance mesure la POLITIQUE et pas la geographie,
    # une meme sonde doit produire une obs IDENTIQUE quel que soit le seed.
    import numpy as np
    o1 = build_probe_obs(make_probe_env(seed=1), "Alone")
    o2 = build_probe_obs(make_probe_env(seed=7), "Alone")
    assert np.array_equal(o1, o2), "biome non neutralise -> obs depend du seed"


def test_token_probes_identical_without_vocab():
    import numpy as np
    env = make_probe_env(seed=1)
    obs0 = build_probe_obs(env, "Token_heard_0")
    obs1 = build_probe_obs(env, "Token_heard_1")
    assert np.array_equal(obs0, obs1)  # sans vocab, heard-embeddings = zero


from aetherlife.viz.policy_probe import fingerprint, policy_distance


def _make_cpu_brain(obs_dim=505, n_actions=9):
    from aetherlife.agents.lineage_brain import BrainConfig, LineageBrain
    cfg = BrainConfig(enabled=True, device="cpu", vision_radius=4,
                      hidden_dims=(64, 64))
    return LineageBrain(root_id=0, obs_dim=obs_dim, n_actions=n_actions, cfg=cfg,
                        seed=0)


def test_fingerprint_shape():
    pytest.importorskip("torch", reason="suite complete : requiert torch")
    pytest.importorskip("mw_ia", reason="suite complete : requiert le repo sibling MW_IA")
    env = make_probe_env(seed=1)
    brain = _make_cpu_brain()
    fp = fingerprint(brain, env)
    assert fp.shape == (len(PROBE_LABELS), 9)
    assert np.isfinite(fp).all()


def test_policy_distance_identical_is_zero():
    fp = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    assert policy_distance(fp, fp) < 1e-9  # cosine de vecteurs identiques ~ 1


def test_policy_distance_symmetric():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert policy_distance(a, b) == policy_distance(b, a)


def test_policy_distance_orthogonal_is_one():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert abs(policy_distance(a, b) - 1.0) < 1e-9


import json


def _write_probe_json(path, seed, mobility, village, fp):
    rec = {
        "seed": seed, "mobility_score": mobility, "village_basin": village,
        "action_labels": ACTION_LABELS, "probe_labels": PROBE_LABELS,
        "fingerprint": fp,
    }
    path.write_text(json.dumps(rec), encoding="utf-8")


def test_render_compare_distance_and_png(tmp_path):
    pytest.importorskip("torch", reason="suite complete : requiert torch")
    pytest.importorskip("mw_ia", reason="suite complete : requiert le repo sibling MW_IA")
    pytest.importorskip("pygame", reason="suite complete : requiert pygame")
    sys.path.insert(0, "scripts")
    from render_policy_compare import compare

    n_p = len(PROBE_LABELS)
    vil = [[1.0] * 9 for _ in range(n_p)]
    mob = [[0.0, 5.0] + [0.0] * 7 for _ in range(n_p)]
    for i in range(3):
        _write_probe_json(tmp_path / f"v{i}.json", i, 0.95, True, vil)
    for i in range(3):
        _write_probe_json(tmp_path / f"m{i}.json", 10 + i, 0.20, False, mob)
    out_png = str(tmp_path / "cmp.png")
    res = compare([str(p) for p in tmp_path.glob("*.json")], out_png)
    assert os.path.getsize(out_png) > 0
    assert res["inter"] > res["intra"]
    assert res["n_village"] == 3 and res["n_mobile"] == 3
