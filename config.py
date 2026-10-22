"""Edit paths here before launching inference.py or test.py from an IDE."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    dataset: str = "Glas"
    data_root: Path = ROOT / "data" / "Glas"
    output_dir: Path = ROOT / "outputs"
    dinov3_repo: Path = ROOT / "external" / "dinov3"
    checkpoint: Path = ROOT / "pretrain" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    encoder_name: str = "dinov3_vitl16"
    image_size: int = 1024
    device: str = "cuda:0"
    seed: int = 42
    save_masks: bool = True

    # Single-image example. Target GT is optional and used for evaluation only.
    reference_image: Path = ROOT / "data" / "Glas" / "train" / "images" / "040.png"
    reference_mask: Path = ROOT / "data" / "Glas" / "train" / "masks" / "040.png"
    target_image: Path = ROOT / "data" / "Glas" / "test" / "images" / "050.png"
    target_mask: Path | None = None


CONFIG = Config()
