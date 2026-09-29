import argparse
import re
import sys
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv

from vpm2.config import Config
from vpm2.context import Context
from vpm2.pipeline import STAGES, run_pipeline
from vpm2.preflight import check_ffmpeg, check_ollama
from vpm2.progress import RichReporter
from vpm2.tts_cache import clear_tts_cache
from vpm2.url import ensure_single_video, sanitize_url


def _resolve_id(url: str) -> str:
    try:
        # extract_flat keeps yt-dlp from recursing into a channel/playlist's
        # videos -- it returns the playlist shell (_type="playlist") instantly,
        # so ensure_single_video can reject it instead of erroring out partway
        # through extracting an entry (e.g. an age-gated video).
        opts = {"quiet": True, "skip_download": True, "extract_flat": "in_playlist"}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:  # noqa: BLE001 - yt-dlp raises many error types; fall back to slug
        info = None
    if info is not None:
        ensure_single_video(url, info)  # fail fast on channels/playlists
        vid = info.get("id")
        if vid:
            return vid
    return re.sub(r"[^A-Za-z0-9_-]", "_", url)[-40:]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # Load .env so HF_TOKEN (and any other secrets) reach the libraries that
    # read them from the process environment -- e.g. huggingface_hub during
    # the transcribe stage.
    load_dotenv()
    ap = argparse.ArgumentParser(prog="vpm2")
    ap.add_argument("url", nargs="?",
                    help="URL do vídeo (opcional só com --clear-tts-cache)")
    ap.add_argument("--voice-mode", choices=["cloning", "preset", "profile"],
                    default="cloning",
                    help="cloning = auto-extract reference from the video; "
                         "preset = use --preset-ref clip; "
                         "profile = reuse a saved voice profile (--voice-profile)")
    ap.add_argument("--voice-profile", default=None,
                    help="saved voice profile id (required when --voice-mode profile)")
    ap.add_argument("--save-profile", default=None,
                    help="persist the auto-extracted reference under this profile id")
    ap.add_argument("--voices-dir", default=None,
                    help="diretório dos perfis de voz (default: voices)")
    ap.add_argument("--asr-backend", choices=["parakeet", "whisper"],
                    default="parakeet")
    ap.add_argument("--preset-ref", default=None,
                    help="clean PT reference wav (required when --voice-mode preset)")
    ap.add_argument("--force", default=None,
                    choices=[s.name for s in STAGES],
                    help="reexecuta a partir deste estágio")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--translate-workers", type=int, default=None,
                    help="concurrent translation requests to Ollama (default 8)")
    ap.add_argument("--asr-model", default=None)
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--tts-cache-dir", default=None,
                    help="diretório do cache de TTS (default: work/_tts_cache)")
    ap.add_argument("--no-tts-cache", action="store_true",
                    help="desliga o cache de TTS entre vídeos")
    ap.add_argument("--clear-tts-cache", action="store_true",
                    help="apaga os clipes do cache de TTS e sai (não roda o pipeline)")
    ap.add_argument("--keep-original-audio", action="store_true",
                    help="keep the English audio as a second track "
                         "(off by default -> output has only the PT-BR dub)")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="max time-stretch for clips that overrun their slot "
                         "(default 1.5; pitch preserved). ffmpeg atempo caps at 2.0")
    args = ap.parse_args(argv)

    # Explicit cache wipe: no URL resolution, no preflight, no pipeline. Keep
    # it a cheap, side-effect-limited op that short-circuits before that work.
    if args.clear_tts_cache:
        cache_dir = args.tts_cache_dir or Config().tts_cache_dir
        removed = clear_tts_cache(cache_dir)
        print(f"[vpm2] cache de TTS limpo: {removed} arquivo(s) removido(s) de {cache_dir}")
        return 0

    if args.url is None:
        ap.error("informe uma URL (ou use --clear-tts-cache)")

    if args.voice_mode == "preset" and not args.preset_ref:
        ap.error("--voice-mode preset requires --preset-ref <clean_pt_voice.wav>")

    config = Config(voice_mode=args.voice_mode, preset_ref_wav=args.preset_ref,
                    keep_original_audio=args.keep_original_audio)
    if args.voice_mode == "profile" and not args.voice_profile:
        ap.error("--voice-mode profile requires --voice-profile <id>")
    config.voice_profile = args.voice_profile
    config.save_profile = args.save_profile
    config.asr_backend = args.asr_backend
    if args.voices_dir:
        config.voices_dir = args.voices_dir
    if args.tts_cache_dir:
        config.tts_cache_dir = args.tts_cache_dir
    if args.no_tts_cache:
        config.tts_cache = False
    if args.max_speed is not None:
        config.max_speed = args.max_speed
    if args.ollama_model:
        config.ollama_model = args.ollama_model
    if args.translate_workers is not None:
        config.translate_workers = args.translate_workers
    if args.asr_model:
        config.asr_model = args.asr_model

    # Fail fast with friendly messages before any heavy work.
    check_ffmpeg()
    check_ollama(config)

    url = sanitize_url(args.url)
    video_id = _resolve_id(url)
    work_dir = Path(args.work_root) / video_id
    reporter = RichReporter()
    ctx = Context(url=url, work_dir=work_dir, config=config, reporter=reporter)

    run_pipeline(ctx, force_from=args.force)
    final = ctx.path("06_final.mp4")
    reporter.console.print(f"\n[bold green]✔ pronto![/bold green] [dim]->[/dim] {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
