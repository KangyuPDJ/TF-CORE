"""Construction is delayed so dataset inspection does not require PyTorch."""

from .modules import make_stages, require_implementations


def build_model(config):
    stages = make_stages()
    # Fail before importing/loading a large encoder when implementations are omitted.
    require_implementations(stages)
    from .backbone import build_encoder
    from .tf_core import TFCORE
    return TFCORE(build_encoder(config), image_size=config.image_size, stages=stages).eval()


__all__ = ["build_model"]
