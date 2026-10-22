"""All method implementations are intentionally omitted.

Fill each forward() method and set implemented=True when releasing the method.
No identity fallback, baseline substitute, or fabricated prediction is provided.
"""


class MissingImplementationError(NotImplementedError):
    pass


class _ModuleInterface:
    implemented = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _omitted(self):
        raise MissingImplementationError(
            f"{type(self).__name__} is an interface only; its implementation was omitted."
        )


class FeaturePreprocessing(_ModuleInterface):
    """Feature preprocessing and clustering interface."""

    def forward(self, reference_features, target_features):
        """[C,Hr,Wr], [C,Ht,Wt] -> FeatureContext."""
        self._omitted()


class InstanceLevelDualCheckPrototypes(_ModuleInterface):
    """IDP interface."""

    def forward(self, context, reference_mask):
        """FeatureContext, bool [Hr,Wr] -> PrototypeBanks."""
        self._omitted()


class EvidenceAccumulationMatching(_ModuleInterface):
    """EAM interface."""

    def forward(self, context, prototypes):
        """FeatureContext, PrototypeBanks -> bool [Ht,Wt]."""
        self._omitted()


class MaximumPoolingDiffusion(_ModuleInterface):
    """MPD interface."""

    def forward(self, context, initial_mask):
        """FeatureContext, bool [Ht,Wt] -> bool [Ht,Wt]."""
        self._omitted()


class BidirectionalVerificationAndCorrection(_ModuleInterface):
    """BVC interface. No target ground-truth mask is accepted."""

    def forward(self, context, reference_mask, candidate_mask):
        """FeatureContext, bool [Hr,Wr], bool [Ht,Wt] -> bool [Ht,Wt]."""
        self._omitted()


class StructureGuidedCompetitiveRefinement(_ModuleInterface):
    """SCR interface."""

    def forward(self, context, candidate_mask):
        """FeatureContext, bool [Ht,Wt] -> bool [Ht,Wt]."""
        self._omitted()


def make_stages():
    return {
        "preprocessing": FeaturePreprocessing(),
        "idp": InstanceLevelDualCheckPrototypes(),
        "eam": EvidenceAccumulationMatching(),
        "mpd": MaximumPoolingDiffusion(),
        "bvc": BidirectionalVerificationAndCorrection(),
        "scr": StructureGuidedCompetitiveRefinement(),
    }


def require_implementations(stages):
    missing = [name for name, stage in stages.items() if not stage.implemented]
    if missing:
        raise MissingImplementationError(
            "Framework only. Implement these interfaces in models/modules.py first: "
            + ", ".join(missing)
        )
