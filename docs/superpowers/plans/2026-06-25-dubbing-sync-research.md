# Research: Audio Sync Strategies for AI Dubbing

> **Status:** Research / to study. This is **not** an approved implementation plan
> yet — it captures the landscape so we can pick the best approach for vpm2 and
> turn the chosen one into a task-by-task plan.

**Problem:** PT-BR dubs run noticeably longer than the English source (measured
~40% longer on a real sample: 463s of TTS audio vs a 326s video). The current
pipeline only stretches audio *after* synthesis, which is the weakest lever and
forces unnatural speed-ups or clips the tail.

**Where we are today:** post-synthesis **time-stretch** in `timeline.py` +
`assemble.py` (ffmpeg `atempo`, pitch preserved). Fixed in this session: clips
that fall behind now compress up to `max_speed` (was silently drifting at 1.0x);
default cap raised 1.25 → 1.5, exposed via `--max-speed`. This makes the dub fit,
but the literature treats stretching as the *last resort* — better approaches act
earlier in the pipeline.

## The 4 families of strategy

### 1. Isochrony / Prosodic alignment — acts in `assemble`/`timeline`
Match the original **phrase-and-pause structure** segment-by-segment instead of
fitting total duration globally. A prosodic-alignment module inserts pauses to
segment the translated text at the same points as the source. On-screen speech
needs stricter alignment; off-screen can relax it.
- **vpm2 fit:** evolve `timeline.py` to align each clip to *its own* original
  slot and distribute pauses, instead of accumulating global drift.
- **Effort:** Medium. **Impact:** High (kills drift; natural rhythm).

### 2. Length control in TRANSLATION — acts in `translate` ⭐ best ROI
Make the MT output **already fit the target duration** (syllable/phoneme budget).
VideoDubber does "speech-aware" length control; isochrony-aware NMT uses auxiliary
counters so the decoder tracks remaining time.
- **vpm2 fit:** we already run an LLM (qwen3) in `translate`. Add a per-segment
  **length constraint** to the prompt (max chars/syllables derived from the
  source segment's duration), so PT-BR comes out concise. Attacks the root cause.
- **Effort:** Low (touches `translate_prompt.py`). **Impact:** High.

### 3. Duration control in SYNTHESIS — acts in `synthesize`
Control duration inside the TTS, not after: unit-based speed adaptation
(Dub-S2ST), or per-phoneme duration via forced alignment (Montreal Forced
Aligner) feeding a TTS duration predictor (FastSpeech-style).
- **vpm2 fit:** Chatterbox exposes no fine duration control, so this needs a
  different TTS backend.
- **Effort:** High (backend swap). **Impact:** High but costly.

### 4. Lip-sync (on-screen) — beyond current scope
True lip synchrony when the mouth is visible. Most expensive; usually unnecessary
for narration/podcast-style content.

## Recommendation (ROI order)

| Priority | Strategy | Where | Effort |
|----------|----------|-------|--------|
| 🥇 | Length control in translation prompt (syllable/char budget per segment) | `translate_prompt.py` | Low |
| 🥈 | Isochronous alignment (clip ↔ its own slot + distribute pauses) | `timeline.py` | Medium |
| 🥉 | Time-stretch (done/improved this session) | `assemble.py` | ✅ |
| — | TTS with duration predictor | new backend | High |

🥇 + 🥈 together would largely remove the need to speed up speech: the
translation comes out short enough and each clip lands in the right place, so
time-stretch becomes a small fine-tune.

## Next step to study

Prototype **#1 (translation length control)** first: compute a per-segment
character/syllable budget from the source segment duration and instruct qwen3 to
respect it, then re-measure total TTS duration vs video. If the overflow drops
near zero, decide whether #2 is still needed.

## Sources

- [Prosodic Alignment for off-screen automatic dubbing (Amazon, arXiv 2204.02530)](https://arxiv.org/abs/2204.02530)
- [Improvements to Prosodic Alignment for Automatic Dubbing (IEEE)](https://ieeexplore.ieee.org/document/9414966)
- [Isochrony-Aware Neural Machine Translation for Automatic Dubbing (arXiv 2112.08548)](https://arxiv.org/pdf/2112.08548)
- [Improving Isochronous MT with Target Factors and Auxiliary Counters (arXiv 2305.13204)](https://arxiv.org/pdf/2305.13204)
- [VideoDubber: MT with Speech-Aware Length Control (arXiv 2211.16934)](https://arxiv.org/abs/2211.16934)
- [Dub-S2ST: Textless Speech-to-Speech Translation for Seamless Dubbing (arXiv 2505.20899)](https://arxiv.org/pdf/2505.20899)
- [Fine-grained Video Dubbing Duration Alignment with SSPO (arXiv 2508.08550)](https://arxiv.org/pdf/2508.08550)
