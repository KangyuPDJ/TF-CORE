"""Local frozen DINOv3 encoder loader. No automatic downloads."""

from pathlib import Path

import torch


def build_encoder(config):
    repository = Path(config.dinov3_repo).expanduser().resolve()
    checkpoint = Path(config.checkpoint).expanduser().resolve()
    if not (repository / "hubconf.py").is_file():
        raise FileNotFoundError(f"Set dinov3_repo to a local DINOv3 repository: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")
    if config.image_size <= 0 or config.image_size % 16:
        raise ValueError("image_size must be a positive multiple of 16.")
    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an available device in config.py.")
    encoder = torch.hub.load(
        str(repository), config.encoder_name, source="local", weights=str(checkpoint)
    )
    return encoder.to(config.device).requires_grad_(False).eval()
