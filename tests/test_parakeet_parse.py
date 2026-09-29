from vpm2.asr.base import RawSegment
from vpm2.asr.parakeet_backend import parse_hypothesis


class FakeHyp:
    def __init__(self, text, timestamp=None):
        self.text = text
        self.timestamp = timestamp


def test_parses_segment_timestamps():
    hyp = FakeHyp("hello world", {"segment": [
        {"segment": "hello", "start": 0.0, "end": 0.5},
        {"segment": "world", "start": 0.5, "end": 1.0},
    ]})
    assert parse_hypothesis(hyp) == [
        RawSegment(0.0, 0.5, "hello"),
        RawSegment(0.5, 1.0, "world"),
    ]


def test_falls_back_to_single_segment_with_audio_duration():
    assert parse_hypothesis(FakeHyp("no timing here"), audio_duration=12.5) == [
        RawSegment(0.0, 12.5, "no timing here"),
    ]


def test_empty_text_yields_no_segments():
    assert parse_hypothesis(FakeHyp("   "), audio_duration=3.0) == []
