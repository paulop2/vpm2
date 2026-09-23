from vpm2.asr.base import RawSegment
from vpm2.asr.normalize import normalize_segments


def test_assigns_sequential_ids_and_preserves_order():
    raw = [RawSegment(0.0, 1.0, "hello"), RawSegment(1.5, 2.0, "world")]
    assert normalize_segments(raw) == [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hello"},
        {"id": 1, "start": 1.5, "end": 2.0, "text": "world"},
    ]


def test_drops_empty_and_whitespace_only_segments():
    raw = [RawSegment(0.0, 1.0, "  "), RawSegment(1.0, 2.0, "ok"), RawSegment(2.0, 3.0, "")]
    out = normalize_segments(raw)
    assert [s["text"] for s in out] == ["ok"]
    assert [s["id"] for s in out] == [0]


def test_sorts_by_start_time():
    raw = [RawSegment(5.0, 6.0, "b"), RawSegment(1.0, 2.0, "a")]
    assert [s["text"] for s in normalize_segments(raw)] == ["a", "b"]
    assert [s["id"] for s in normalize_segments(raw)] == [0, 1]


def test_strips_text():
    raw = [RawSegment(0.0, 1.0, "  hi  ")]
    assert normalize_segments(raw)[0]["text"] == "hi"


def test_empty_input_returns_empty_list():
    assert normalize_segments([]) == []
