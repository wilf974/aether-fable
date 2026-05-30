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
