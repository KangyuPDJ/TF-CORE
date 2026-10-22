"""Optional CLI overrides; config.py remains the central configuration."""

import argparse
from dataclasses import replace
from pathlib import Path

from config import CONFIG


def get_args_parser():
    parser = argparse.ArgumentParser(description="TF-CORE / GlaS framework only")
    parser.add_argument("--dataset", choices=["Glas"], default=CONFIG.dataset)
    parser.add_argument("--data-root", type=Path, default=CONFIG.data_root)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--dinov3-repo", type=Path, default=CONFIG.dinov3_repo)
    parser.add_argument("--checkpoint", type=Path, default=CONFIG.checkpoint)
    parser.add_argument("--image-size", type=int, default=CONFIG.image_size)
    parser.add_argument("--device", default=CONFIG.device)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--check-data", action="store_true",
                        help="Validate all images/masks without loading a model.")
    return parser


def parse_config(argv=None):
    options = vars(get_args_parser().parse_args(argv))
    check_data = options.pop("check_data")
    return replace(CONFIG, **options), check_data
