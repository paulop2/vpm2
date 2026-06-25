import json
from pathlib import Path

from vpm2.artifacts import (
    read_json, write_json,
    valid_transcript, valid_translation, valid_clips,
)


def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"a": 1})
    assert read_json(p) == {"a": 1}


def test_valid_transcript_true(tmp_path):
    p = tmp_path / "t.json"
    write_json(p, {"language": "en", "segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hi"}]})
    assert valid_transcript(p) is True


def test_valid_transcript_false_when_empty(tmp_path):
    p = tmp_path / "t.json"
    write_json(p, {"language": "en", "segments": []})
    assert valid_transcript(p) is False


def test_valid_transcript_false_when_missing_file(tmp_path):
    assert valid_transcript(tmp_path / "nope.json") is False


def test_valid_translation_requires_text_pt(tmp_path):
    p = tmp_path / "tr.json"
    write_json(p, {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "text_pt": ""}]})
    assert valid_translation(p) is False
    write_json(p, {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "text_pt": "oi"}]})
    assert valid_translation(p) is True


def test_valid_clips_checks_files_exist(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "0000.wav").write_bytes(b"RIFF")
    p = tmp_path / "c.json"
    write_json(p, {"sample_rate": 24000, "segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "clip": "0000.wav", "duration": 0.9}]})
    assert valid_clips(p, clips) is True
    # missing file -> invalid
    write_json(p, {"sample_rate": 24000, "segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "clip": "9999.wav", "duration": 0.9}]})
    assert valid_clips(p, clips) is False