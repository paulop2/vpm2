"""URL hygiene for the CLI: undo shell-escape artifacts and reject non-videos."""


def sanitize_url(url: str) -> str:
    r"""Strip shell-escape artifacts from a pasted URL.

    A backslash is never valid in a YouTube URL, but a URL pasted inside double
    quotes as ``"watch\?v\=ID"`` keeps its literal backslashes -- the shell only
    unescapes ``\`` before ``$``, ``` ` ```, ``"`` and ``\`` itself, so ``\?``
    and ``\=`` survive. Those break URL parsing and make yt-dlp fall back to the
    channel/tab extractor (downloading the wrong thing). Drop the backslashes and
    surrounding whitespace.
    """
    return url.replace("\\", "").strip()


def ensure_single_video(url: str, info: dict) -> None:
    """Raise if yt-dlp resolved *url* to a channel/playlist instead of a video.

    Channels and playlists come back from the ``youtube:tab`` extractor as a
    ``_type == "playlist"`` entry carrying ``entries``; a single video has
    neither. Failing here keeps a mistyped URL from silently grabbing a whole
    channel and dying downstream on a missing ``01_video.mp4``.
    """
    if info.get("_type") == "playlist" or info.get("entries") is not None:
        raise SystemExit(
            f"[vpm2] '{url}' resolveu para um canal/playlist, não um vídeo único. "
            "Passe a URL de um vídeo (https://www.youtube.com/watch?v=ID); "
            "e não escape ? e = quando a URL estiver entre aspas."
        )
