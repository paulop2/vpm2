from vpm2.voice_sample import pick_reference_window


def test_picks_inside_longest_span():
    spans = [(0.0, 2.0), (10.0, 30.0), (31.0, 33.0)]
    win = pick_reference_window(spans, target=10.0, min_len=6.0)
    assert win is not None
    start, end = win
    assert 10.0 <= start < end <= 30.0
    assert end - start <= 10.0
    assert end - start >= 6.0


def test_returns_none_when_no_span_long_enough():
    spans = [(0.0, 2.0), (5.0, 9.0)]
    assert pick_reference_window(spans, target=10.0, min_len=6.0) is None


def test_window_clamped_to_span_end():
    spans = [(0.0, 7.0)]  # 7s span, target 10 -> clamp to span end
    win = pick_reference_window(spans, target=10.0, min_len=6.0)
    assert win is not None
    start, end = win
    assert end <= 7.0