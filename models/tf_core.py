"""TF-CORE call graph and generic image handling; research methods are omitted."""

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from utils.io import binary_array, read_binary_mask, read_rgb
from .modules import make_stages, require_implementations


class TFCORE(nn.Module):
    def __init__(self, encoder, image_size=1024, stages=None):
        super().__init__()
        if image_size <= 0 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16.")
        self.encoder = encoder.requires_grad_(False).eval()
        self.image_size = image_size
        self.stages = make_stages() if stages is None else stages
        expected = set(make_stages())
        if set(self.stages) != expected:
            raise ValueError(f"Expected exactly these stage names: {sorted(expected)}")
        self._reference = self._reference_mask = self._target = None

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def set_reference(self, image, mask):
        self._reference = read_rgb(image)
        if isinstance(mask, (str, bytes)) or hasattr(mask, "__fspath__"):
            self._reference_mask = read_binary_mask(mask)
        else:
            self._reference_mask = binary_array(mask)
        if self._reference_mask.shape != (self._reference.height, self._reference.width):
            raise ValueError("Reference image and mask must have identical spatial sizes.")
        if not self._reference_mask.any():
            raise ValueError("The reference mask must contain foreground.")

    def set_target(self, image):
        self._target = read_rgb(image)

    def _image_tensor(self, image):
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def _features(self, image):
        features = self.encoder.get_intermediate_layers(
            self._image_tensor(image), n=1, reshape=True
        )[0]
        if features.ndim != 4 or features.shape[0] != 1:
            raise ValueError("The encoder must return [1,C,Hf,Wf] patch features.")
        return features[0]

    @torch.inference_mode()
    def segment(self):
        require_implementations(self.stages)
        if self._reference is None or self._target is None:
            raise RuntimeError("Call set_reference() and set_target() before segment().")
        reference_features = self._features(self._reference)
        target_features = self._features(self._target)
        reference_mask = torch.from_numpy(self._reference_mask.copy()).to(self.device)
        reference_mask = F.interpolate(
            reference_mask[None, None].float(), size=reference_features.shape[-2:],
            mode="nearest",
        )[0, 0].bool()
        if not reference_mask.any():
            raise ValueError("Reference foreground disappears at the feature-map resolution.")

        context = self.stages["preprocessing"](reference_features, target_features)
        prototypes = self.stages["idp"](context, reference_mask)
        initial = self.stages["eam"](context, prototypes)
        diffused = self.stages["mpd"](context, initial)
        corrected = self.stages["bvc"](context, reference_mask, diffused)
        refined = self.stages["scr"](context, corrected)

        mask = torch.as_tensor(binary_array(refined), device=self.device)
        if tuple(mask.shape) != tuple(target_features.shape[-2:]):
            raise ValueError("The final stage must return a binary mask at target-feature resolution.")
        return F.interpolate(
            mask[None, None].float(), size=(self._target.height, self._target.width),
            mode="nearest",
        )[0, 0].bool()
