"""Input loading and annotation-free binary prediction export."""

from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(source):
    if isinstance(source, Image.Image):
        return source.convert("RGB").copy()
    with Image.open(source) as image:
        return image.convert("RGB").copy()


def read_binary_mask(path):
    with Image.open(path) as image:
        # Preserve indexed/16-bit instance IDs; an RGB foreground uses any nonzero channel.
        array = np.asarray(image)
        if array.ndim == 3:
            array = np.any(array[..., :3] > 0, axis=-1)
        return np.array(array > 0, dtype=bool, copy=True)


def binary_array(mask):
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D binary mask, got shape {array.shape}.")
    if not np.isin(array, [0, 1]).all():
        raise ValueError("Predictions must contain only 0/1, not probabilities or arbitrary values.")
    return array.astype(bool, copy=False)


def save_binary_mask(mask, path):
    output = Path(path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_array(mask).astype(np.uint8) * 255).save(output, format="PNG")
    return output
