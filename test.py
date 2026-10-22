"""Single-sample entry point; configure image paths in config.py."""

import sys

from config import CONFIG
from models import build_model
from models.modules import MissingImplementationError
from utils.io import read_binary_mask, save_binary_mask
from utils.metrics import Confusion, format_scores
from utils.runtime import seed_everything


def main(config=CONFIG):
    try:
        model = build_model(config)
    except MissingImplementationError as error:
        print(str(error), file=sys.stderr)
        return 2
    seed_everything(config.seed)
    model.set_reference(config.reference_image, config.reference_mask)
    model.set_target(config.target_image)
    prediction = model.segment()
    output = save_binary_mask(prediction, config.output_dir / "single" / "prediction.png")
    print(f"Binary mask: {output}")
    if config.target_mask is not None:
        meter = Confusion()
        meter.update(prediction, read_binary_mask(config.target_mask))
        print(format_scores(meter.scores()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
