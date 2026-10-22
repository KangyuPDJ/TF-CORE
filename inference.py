"""GlaS batch-evaluation entry point. No method implementations are included."""

import csv
from dataclasses import asdict
import json
import sys

from datasets import build_dataset
from models import build_model
from models.modules import MissingImplementationError
from opts import parse_config
from utils.io import save_binary_mask
from utils.metrics import Confusion, format_scores
from utils.runtime import seed_everything


def evaluate(config, model, dataset):
    """Target GT enters only the evaluator, never the prediction API."""
    rows = []
    total = Confusion()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for index in range(len(dataset)):
        sample = dataset[index]
        model.set_reference(sample["ref_imgs"][0], sample["ref_masks"][0])
        model.set_target(sample["tgt_img"])
        prediction = model.segment()
        item = Confusion()
        item.update(prediction, sample["tgt_mask"])
        total.update(prediction, sample["tgt_mask"])
        mask_path = ""
        if config.save_masks:
            mask_path = str(save_binary_mask(prediction, output / "masks" / sample["tgt_name"]))
        rows.append({
            "target": sample["tgt_image_path"], "reference": sample["ref_image_path"],
            "prediction": mask_path, **asdict(item), **item.scores(),
        })
        print(f"[{index + 1}/{len(dataset)}] {sample['tgt_name']}: {format_scores(item.scores())}")

    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": "Glas", "seed": config.seed,
        "reference_count": len(dataset.references), "target_count": len(dataset),
        "aggregation": "foreground metrics from pooled test-pixel counts",
        "counts": asdict(total), "metrics_percent": total.scores(),
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Final: " + format_scores(total.scores()))
    return summary


def main(argv=None):
    config, check_data = parse_config(argv)
    dataset = build_dataset(config.dataset, config)
    if check_data:
        dataset.validate()
        print(f"GlaS data checked: {len(dataset.references)} references; {len(dataset)} targets.")
        return 0
    try:
        model = build_model(config)
    except MissingImplementationError as error:
        print(str(error), file=sys.stderr)
        return 2
    seed_everything(config.seed)
    evaluate(config, model, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
