import pytest

from vpm2.asr.base import get_asr_backend
from vpm2.config import Config


def test_unknown_backend_raises_without_importing_models():
    with pytest.raises(ValueError, match="unknown asr backend"):
        get_asr_backend(Config(asr_backend="nope"))


def test_whisper_backend_selected_by_config():
    assert type(get_asr_backend(Config(asr_backend="whisper"))).__name__ == "WhisperBackend"


def test_parakeet_is_default_backend():
    assert type(get_asr_backend(Config())).__name__ == "ParakeetBackend"
