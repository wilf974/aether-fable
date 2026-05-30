import json
from aetherlife.viz.v8_replay import token_color, lineage_color, TOKEN_COLORS
from aetherlife.viz.v8_replay import (
    load_meta, iter_events, validate_event,
)


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
