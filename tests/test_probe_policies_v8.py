"""Smoke capture OBS V3.0 — probe_policies_v8 produit un JSON valide."""
import json
import pytest


def test_probe_capture_produces_valid_json(tmp_path):
    pytest.importorskip("torch", reason="suite complete : requiert torch")
    import os
    import subprocess
    import sys

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
