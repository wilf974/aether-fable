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
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["total_ticks"] == 40 and meta["record_every"] == 10
    assert meta["schema_version"] == 2
    with open(os.path.join(out_dir, "events.jsonl"), encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    assert len(lines) >= 1
    ev = lines[0]
    assert {"t", "agents", "vocal", "spots", "n_alive", "food"} <= set(ev.keys())
    assert len(ev["food"]) == 64  # grille 8x8 food par région (schema v2)
    if ev["agents"]:
        a = ev["agents"][0]
        assert {"id", "lin", "r", "c", "e", "er", "age", "aff"} <= set(a.keys())
