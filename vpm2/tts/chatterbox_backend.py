from pathlib import Path

import numpy as np

from vpm2.config import Config
from vpm2.tts.base import TTSBackend


class ChatterboxBackend(TTSBackend):
    def __init__(self, config: Config):
        self._config = config
        self._model = None
        self._prepared_ref: str | None = None

    def _ensure_model(self):
        if self._model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self._model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
            self.sample_rate = int(self._model.sr)

    def synth(self, text: str, ref_wav: Path | None) -> np.ndarray:
        self._ensure_model()
        # The reference voice is the same for every segment of a video, but
        # generate(audio_prompt_path=...) re-embeds it on each call. Prepare the
        # conditionals once and reuse them -- generate() then reads self.conds.
        if ref_wav is not None and str(ref_wav) != self._prepared_ref:
            self._model.prepare_conditionals(str(ref_wav))
            self._prepared_ref = str(ref_wav)
        wav = self._model.generate(text, language_id=self._config.target_lang)
        arr = wav.squeeze().detach().cpu().numpy().astype("float32")
        return arr
