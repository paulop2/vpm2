def free_cuda() -> None:
    # Stages run sequentially and each holds the whole GPU to itself. The caching
    # allocator keeps freed blocks reserved, so without this the next stage sees
    # a VRAM floor it never used -- on a single 16GB card that's the difference
    # between Whisper + Chatterbox coexisting or OOMing. Call after a GPU stage
    # has dropped its model references.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # No torch / no CUDA build (e.g. CI): nothing to free.
        pass
