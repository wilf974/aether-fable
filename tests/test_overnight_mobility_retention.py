"""Intégration — overnight_v8b1 retient l'occupation (chantier A, observation-only).

Vérifie que run_overnight injecte le bloc spatial_mobility_v8c3 dans le report,
sans rien changer d'autre. Run CPU très court.
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)


def test_run_overnight_emits_spatial_mobility_block(tmp_path):
    from overnight_v8b1 import run_overnight

    report = run_overnight(
        n_ticks=60, seed=1, device="cpu",
        out_dir=str(tmp_path), regime="coordination_collective",
    )
    assert "spatial_mobility_v8c3" in report
    sm = report["spatial_mobility_v8c3"]
    assert {
        "corr_occupation_start_end", "village_basin",
        "start_window_ticks", "end_window_ticks",
        "n_samples_start", "n_samples_end",
    } <= set(sm)
    # fenêtre = 10 % de 60 = 6 ticks
    assert sm["start_window_ticks"] == [0, 6]
    assert sm["end_window_ticks"] == [54, 60]
    assert sm["n_samples_start"] > 0
