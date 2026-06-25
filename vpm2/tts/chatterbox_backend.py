from pathlib import Path

import numpy as np

from vpm2.config import Config
from vpm2.tts.base import TTSBackend


class ChatterboxBackend(TTSBackend):
    def __init__(self, config: Config):
        self._config = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self._model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
            self.sample_rate = int(self._model.sr)

    def synth(self, text: str, ref_wav: Path | None) -> np.ndarray:
        self._ensure_model()
        kwargs = {"language_id": self._config.target_lang}
        if ref_wav is not None:
            kwargs["audio_prompt_path"] = str(ref_wav)
        wav = self._model.generate(text, **kwargs)
        arr = wav.squeeze().detach().cpu().numpy().astype("float32")
        return arr
