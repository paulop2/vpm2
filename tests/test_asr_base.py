import pytest

from vpm2.asr.base import get_asr_backend
from vpm2.config import Config


def test_unknown_backend_raises_without_importing_models():
    with pytest.raises(ValueError, match="unknown asr backend"):
        get_asr_backend(Config(asr_backend="nope"))
