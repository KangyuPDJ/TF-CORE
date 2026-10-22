"""One-way one-shot GlaS episodes: train references and exhaustive test targets."""

from pathlib import Path
import random

from utils.io import read_binary_mask, read_rgb

IMAGE_SUFFIXES = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def _index_files(directory, is_mask=False):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing GlaS directory: {directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.stem
        if is_mask and key.endswith("_anno"):
            key = key[:-5]
        if key in result:
            raise ValueError(f"Duplicate image/mask stem '{key}' in {directory}")
        result[key] = path
    if not result:
        raise ValueError(f"No images/masks found in {directory}")
    return result


def _collect_pairs(root, split):
    images = _index_files(root / split / "images")
    masks = _index_files(root / split / "masks", is_mask=True)
    if images.keys() != masks.keys():
        missing = sorted(images.keys() - masks.keys())
        orphan = sorted(masks.keys() - images.keys())
        raise ValueError(f"{split}: images without masks={missing}; masks without images={orphan}")
    return [(images[key], masks[key]) for key in sorted(images)]


class DatasetGland:
    """Masks may be binary, instance-labelled, or RGB; any nonzero pixel is FG.

    Reference selection depends only on (seed, target index), not access order.
    data_root may point to Glas itself or its immediate parent.
    """

    class_ids = (0,)

    def __init__(self, datapath, shot=1, seed=42):
        if shot != 1:
            raise ValueError("This skeleton supports only the 1-way 1-shot protocol.")
        root = Path(datapath).expanduser().resolve()
        if not (root / "train").is_dir() and (root / "Glas").is_dir():
            root = root / "Glas"
        self.root = root
        self.seed = seed
        self.references = _collect_pairs(root, "train")
        self.targets = _collect_pairs(root, "test")
        reference_paths = {p.resolve() for pair in self.references for p in pair}
        target_paths = {p.resolve() for pair in self.targets for p in pair}
        if reference_paths & target_paths:
            raise ValueError("Reference and target files must not resolve to the same paths.")

    def __len__(self):
        return len(self.targets)

    def validate(self):
        """Read every reference and target pair, not only sampled references."""
        for split, pairs in (("train", self.references), ("test", self.targets)):
            for image_path, mask_path in pairs:
                image, mask = read_rgb(image_path), read_binary_mask(mask_path)
                if mask.shape != (image.height, image.width):
                    raise ValueError(f"Image/mask size mismatch: {image_path}")
                if split == "train" and not mask.any():
                    raise ValueError(f"Reference mask has no foreground: {mask_path}")

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ref_index = random.Random(self.seed + index).randrange(len(self.references))
        ref_image_path, ref_mask_path = self.references[ref_index]
        tgt_image_path, tgt_mask_path = self.targets[index]
        reference, ref_mask = read_rgb(ref_image_path), read_binary_mask(ref_mask_path)
        target, tgt_mask = read_rgb(tgt_image_path), read_binary_mask(tgt_mask_path)
        if ref_mask.shape != (reference.height, reference.width):
            raise ValueError(f"Reference image/mask size mismatch: {ref_image_path}")
        if tgt_mask.shape != (target.height, target.width):
            raise ValueError(f"Target image/mask size mismatch: {tgt_image_path}")
        if not ref_mask.any():
            raise ValueError(f"Reference mask has no foreground: {ref_mask_path}")
        return {
            "ref_imgs": [reference], "ref_masks": [ref_mask],
            "tgt_img": target, "tgt_mask": tgt_mask, "class_id": 0,
            "tgt_name": tgt_image_path.stem,
            "ref_image_path": str(ref_image_path), "ref_mask_path": str(ref_mask_path),
            "tgt_image_path": str(tgt_image_path), "tgt_mask_path": str(tgt_mask_path),
        }
