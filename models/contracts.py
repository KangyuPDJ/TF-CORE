"""Input/output containers only; no segmentation algorithm is included."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class FeatureContext:
    reference_features: "Tensor"  # [C, Hr, Wr]
    target_features: "Tensor"     # [C, Ht, Wt]
    reference_clusters: "Tensor"  # [Hr, Wr]
    target_clusters: "Tensor"     # [Ht, Wt]


@dataclass
class PrototypeBanks:
    foreground: "Tensor"         # [K_fg, C]
    background: "Tensor"         # [K_bg, C]
