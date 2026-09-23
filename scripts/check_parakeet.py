import sys
from pathlib import Path

from vpm2.asr.parakeet_backend import ParakeetBackend
from vpm2.config import Config


def main() -> None:
    audio = Path(sys.argv[1])
    segments = ParakeetBackend(Config()).transcribe(audio)
    for s in segments:
        print(f"[{s.start:7.2f} - {s.end:7.2f}] {s.text}")


if __name__ == "__main__":
    main()
