"""Only GlaS is included in this package."""

from .Glas import DatasetGland


def build_dataset(dataset, args):
    if dataset != "Glas":
        raise ValueError("This framework includes only the Glas dataset.")
    return DatasetGland(args.data_root, shot=1, seed=args.seed)


__all__ = ["DatasetGland", "build_dataset"]
