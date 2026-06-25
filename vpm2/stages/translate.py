import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

from vpm2.artifacts import read_json, valid_translation, write_json
from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.translate_prompt import build_translation_prompt


class TranslateStage(Stage):
    name = "translate"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("04_translation.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_translation(self.output_path(ctx))

    def _translate_one(self, ctx, text, prev, nxt):
        prompt = build_translation_prompt(text, prev, nxt)
        resp = requests.post(
            f"{ctx.config.ollama_url}/api/generate",
            json={"model": ctx.config.ollama_model,
                  "prompt": prompt, "stream": False,
                  "think": False,
                  "options": {"temperature": 0.3}},
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["response"]
        return _THINK_RE.sub("", text).strip()

    def run(self, ctx: Context) -> None:
        data = read_json(ctx.path("03_transcript.json"))
        segs = data["segments"]

        def translate_at(i: int) -> str:
            prev = segs[i - 1]["text"] if i > 0 else None
            nxt = segs[i + 1]["text"] if i + 1 < len(segs) else None
            return self._translate_one(ctx, segs[i]["text"], prev, nxt)

        # Fan out across Ollama; segments are independent so order doesn't matter
        # for correctness -- we slot each result back by its original index.
        texts_pt: list[str | None] = [None] * len(segs)
        workers = max(1, min(ctx.config.translate_workers, len(segs) or 1))
        with ctx.reporter.bar("traduzindo (EN→PT-BR)", total=len(segs)) as bar:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(translate_at, i): i for i in range(len(segs))}
                for fut in as_completed(futures):
                    texts_pt[futures[fut]] = fut.result()
                    bar.advance()
        out = [{**s, "text_pt": texts_pt[i]} for i, s in enumerate(segs)]
        write_json(self.output_path(ctx), {"segments": out})
