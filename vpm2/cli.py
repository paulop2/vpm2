import argparse
import re
import sys
from pathlib import Path

import yt_dlp

from vpm2.config import Config
from vpm2.context import Context
from vpm2.pipeline import run_pipeline
from vpm2.preflight import check_ffmpeg, check_ollama
from vpm2.url import ensure_single_video, sanitize_url


def _resolve_id(url: str) -> str:
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        info = None
    if info is not None:
        ensure_single_video(url, info)  # fail fast on channels/playlists
        vid = info.get("id")
        if vid:
            return vid
    return re.sub(r"[^A-Za-z0-9_-]", "_", url)[-40:]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="vpm2")
    ap.add_argument("url")
    ap.add_argument("--voice-mode", choices=["cloning", "preset"], default="cloning",
                    help="cloning = auto-extract reference from the video (zero-config); "
                         "preset = use --preset-ref clip")
    ap.add_argument("--preset-ref", default=None,
                    help="clean PT reference wav (required when --voice-mode preset)")
    ap.add_argument("--force", default=None,
                    help="rerun from this stage name onward")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--asr-model", default=None)
    ap.add_argument("--work-root", default="work")
    args = ap.parse_args(argv)

    if args.voice_mode == "preset" and not args.preset_ref:
        ap.error("--voice-mode preset requires --preset-ref <clean_pt_voice.wav>")

    config = Config(voice_mode=args.voice_mode, preset_ref_wav=args.preset_ref)
    if args.ollama_model:
        config.ollama_model = args.ollama_model
    if args.asr_model:
        config.asr_model = args.asr_model

    # Fail fast with friendly messages before any heavy work.
    check_ffmpeg()
    check_ollama(config)

    url = sanitize_url(args.url)
    video_id = _resolve_id(url)
    work_dir = Path(args.work_root) / video_id
    ctx = Context(url=url, work_dir=work_dir, config=config)

    run_pipeline(ctx, force_from=args.force)
    final = ctx.path("06_final.mp4")
    print(f"[vpm2] done -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
