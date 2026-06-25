import shutil

import requests

from vpm2.config import Config


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "[vpm2] ffmpeg não encontrado no PATH. "
            "Instale com: sudo apt install ffmpeg"
        )


def check_ollama(config: Config) -> None:
    try:
        resp = requests.get(f"{config.ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        raise SystemExit(
            f"[vpm2] Ollama não respondeu em {config.ollama_url}. "
            "Inicie o Ollama (`ollama serve`) antes de rodar."
        )
    names = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(config.ollama_model in n for n in names):
        raise SystemExit(
            f"[vpm2] modelo '{config.ollama_model}' não está disponível no Ollama. "
            f"Rode: ollama pull {config.ollama_model}"
        )
