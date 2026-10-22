"""Synthetic I/O/metric tests only. No research method is implemented here."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from importlib.util import find_spec
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from config import CONFIG
from datasets import build_dataset
from datasets.Glas import DatasetGland
from inference import evaluate, main
from models import build_model
from models.modules import MissingImplementationError, make_stages
from utils.io import binary_array, read_binary_mask, save_binary_mask
from utils.metrics import Confusion


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "Glas"
        for split, count in (("train", 3), ("test", 2)):
            for subdirectory in ("images", "masks"):
                (self.root / split / subdirectory).mkdir(parents=True)
            for index in range(count):
                image = np.full((7, 9, 3), index * 30, dtype=np.uint8)
                mask = np.zeros((7, 9), dtype=np.uint8)
                mask[1:5, 2:6] = 255
                Image.fromarray(image).save(self.root / split / "images" / f"{index:03}.png")
                Image.fromarray(mask).save(self.root / split / "masks" / f"{index:03}.png")

    def test_test_set_is_exhaustive_and_reference_sampling_is_stable(self):
        first = DatasetGland(self.root, seed=42)
        second = DatasetGland(self.root, seed=42)
        self.assertEqual(len(first), 2)
        self.assertEqual([first[i]["tgt_name"] for i in range(len(first))], ["000", "001"])
        first[1]
        self.assertEqual(first[0]["ref_image_path"], second[0]["ref_image_path"])
        for index in range(len(first)):
            self.assertIn("train", Path(first[index]["ref_image_path"]).parts)
            self.assertIn("test", Path(first[index]["tgt_image_path"]).parts)
        with self.assertRaises(IndexError):
            first[len(first)]

    def test_only_glas_is_registered(self):
        with self.assertRaises(ValueError):
            build_dataset("OtherDataset", CONFIG)
        self.assertEqual(len(DatasetGland(self.root.parent)), 2)
        with self.assertRaises(ValueError):
            DatasetGland(self.root, shot=2)

    def test_missing_mask_is_rejected(self):
        (self.root / "test" / "masks" / "000.png").unlink()
        with self.assertRaisesRegex(ValueError, "images without masks"):
            DatasetGland(self.root)

    def test_annotation_suffix_and_bmp_are_supported(self):
        source = self.root / "test" / "masks" / "000.png"
        with Image.open(source) as mask:
            mask.save(source.with_name("000_anno.bmp"))
        source.unlink()
        dataset = DatasetGland(self.root)
        self.assertEqual(dataset[0]["tgt_mask"].dtype, np.bool_)

    def test_all_references_are_validated(self):
        invalid = self.root / "train" / "masks" / "002.png"
        Image.fromarray(np.zeros((7, 9), dtype=np.uint8)).save(invalid)
        with self.assertRaisesRegex(ValueError, "no foreground"):
            DatasetGland(self.root).validate()

    def test_rgb_instance_and_indexed_masks(self):
        path = Path(self.temporary.name) / "mask.png"
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[0, 1, 1] = 13
        Image.fromarray(rgb).save(path)
        self.assertEqual(int(read_binary_mask(path).sum()), 1)
        indexed = Image.fromarray(np.array([[0, 2], [1, 0]], dtype=np.uint8)).convert("P")
        indexed.save(path)
        self.assertEqual(int(read_binary_mask(path).sum()), 2)

    def test_masks_are_saved_as_binary_png_even_without_extension(self):
        mask = np.array([[False, True], [True, False]])
        output = save_binary_mask(mask, Path(self.temporary.name) / "nested" / "result")
        self.assertEqual(output.suffix, ".png")
        with Image.open(output) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.format, "PNG")
            self.assertEqual(set(np.unique(image)), {0, 255})
            np.testing.assert_array_equal(np.array(image) > 0, mask)

    def test_probabilities_are_not_silently_treated_as_binary(self):
        with self.assertRaises(ValueError):
            binary_array(np.array([[0.1, 0.9]]))
        with self.assertRaises(ValueError):
            binary_array(np.ones((1, 2, 3)))

    def test_metrics_match_confusion_counts(self):
        prediction = np.array([[1, 1], [0, 0]])
        ground_truth = np.array([[1, 0], [1, 0]])
        meter = Confusion()
        meter.update(prediction, ground_truth)
        self.assertEqual((meter.tp, meter.fp, meter.fn, meter.tn), (1, 1, 1, 1))
        self.assertAlmostEqual(meter.scores()["mIoU"], 100 / 3)
        self.assertAlmostEqual(meter.scores()["Precision"], 50)
        self.assertAlmostEqual(meter.scores()["Recall"], 50)
        self.assertAlmostEqual(meter.scores()["F1"], 50)
        self.assertTrue(all(value == 0 for value in Confusion().scores().values()))
        with self.assertRaises(ValueError):
            meter.update(np.ones((1, 2)), np.ones((2, 1)))

    def test_all_algorithm_interfaces_raise(self):
        for stage in make_stages().values():
            self.assertFalse(stage.implemented)
            with self.assertRaises(MissingImplementationError):
                stage._omitted()
        with self.assertRaisesRegex(MissingImplementationError, "idp, eam, mpd, bvc, scr"):
            build_model(CONFIG)

    def test_cli_data_check_and_missing_implementation_exit(self):
        captured = StringIO()
        with redirect_stdout(captured):
            code = main(["--data-root", str(self.root), "--check-data"])
        self.assertEqual(code, 0)
        self.assertIn("3 references; 2 targets", captured.getvalue())
        with redirect_stderr(captured):
            code = main(["--data-root", str(self.root)])
        self.assertEqual(code, 2)
        self.assertIn("Framework only", captured.getvalue())

    def test_evaluation_writes_predictions_and_metrics_without_gt_in_model_api(self):
        # This all-foreground dummy is confined to tests. It is not a TF-CORE baseline.
        class DummyModel:
            def set_reference(self, image, mask):
                pass

            def set_target(self, image):
                self.size = (image.height, image.width)

            def segment(self):
                return np.ones(self.size, dtype=bool)

        configuration = replace(CONFIG, data_root=self.root,
                                output_dir=Path(self.temporary.name) / "results")
        with redirect_stdout(StringIO()):
            summary = evaluate(configuration, DummyModel(), DatasetGland(self.root))
        self.assertEqual(summary["target_count"], 2)
        self.assertEqual(len(list((configuration.output_dir / "masks").glob("*.png"))), 2)
        self.assertTrue((configuration.output_dir / "per_image.csv").is_file())
        with (configuration.output_dir / "metrics.json").open(encoding="utf-8") as handle:
            self.assertAlmostEqual(json.load(handle)["metrics_percent"]["Recall"], 100)

    @unittest.skipUnless(find_spec("torch") is not None, "PyTorch is not installed")
    def test_pipeline_call_order_with_test_only_stubs(self):
        import torch
        from models.contracts import FeatureContext
        from models.tf_core import TFCORE

        order = []

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()))

            def get_intermediate_layers(self, tensor, n, reshape):
                return [torch.ones((1, 3, 4, 4), device=tensor.device)]

        class Stub:
            implemented = True

            def __init__(self, name):
                self.name = name

            def __call__(self, *arguments):
                order.append(self.name)
                if self.name == "preprocessing":
                    return FeatureContext(*arguments, torch.zeros((4, 4)), torch.zeros((4, 4)))
                if self.name == "idp":
                    return None
                return torch.ones((4, 4), dtype=torch.bool)

        stages = {name: Stub(name) for name in make_stages()}
        model = TFCORE(Encoder(), image_size=64, stages=stages)
        dataset = DatasetGland(self.root)
        sample = dataset[0]
        model.set_reference(sample["ref_imgs"][0], sample["ref_masks"][0])
        model.set_target(sample["tgt_img"])
        output = model.segment()
        self.assertEqual(tuple(output.shape), (7, 9))
        self.assertEqual(output.dtype, torch.bool)
        self.assertEqual(order, list(make_stages()))
        self.assertFalse(next(model.encoder.parameters()).requires_grad)


if __name__ == "__main__":
    unittest.main()
