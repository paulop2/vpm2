import numpy as np
import soundfile as sf

from vpm2.asr.base import RawSegment
from vpm2.asr.parakeet_backend import (
    ParakeetBackend, offset_segments, parse_hypothesis, plan_chunks,
)
from vpm2.config import Config


class FakeHyp:
    def __init__(self, text, timestamp=None):
        self.text = text
        self.timestamp = timestamp


class FakeModel:
    def __init__(self, sr=16000):
        self.sr = sr
        self.chunk_lengths = []

    def transcribe(self, audios, timestamps=True):
        self.chunk_lengths.append(audios[0].shape[0])
        n = audios[0].shape[0] / self.sr
        return [FakeHyp("x", {"segment": [
            {"segment": "x", "start": 0.0, "end": n}]})]


def test_plan_chunks_splits_evenly_and_last_is_short():
    assert plan_chunks(250.0, 120) == [(0.0, 120.0), (120.0, 240.0), (240.0, 250.0)]


def test_plan_chunks_single_when_shorter_than_chunk():
    assert plan_chunks(30.0, 120) == [(0.0, 30.0)]


def test_offset_segments_shifts_times():
    segs = [RawSegment(0.0, 1.0, "a"), RawSegment(2.0, 3.0, "b")]
    assert offset_segments(segs, 120.0) == [
        RawSegment(120.0, 121.0, "a"),
        RawSegment(122.0, 123.0, "b"),
    ]


def test_transcribe_chunks_and_offsets(tmp_path):
    sr = 16000
    src = tmp_path / "long.wav"
    sf.write(str(src), np.zeros(sr * 3, dtype="float32"), sr)

    backend = ParakeetBackend(Config(parakeet_chunk_seconds=1))
    backend._model = FakeModel(sr=sr)

    segs = backend.transcribe(src)

    assert backend._model.chunk_lengths == [sr, sr, sr]
    assert segs == [
        RawSegment(0.0, 1.0, "x"),
        RawSegment(1.0, 2.0, "x"),
        RawSegment(2.0, 3.0, "x"),
    ]
