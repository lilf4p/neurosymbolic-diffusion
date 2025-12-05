"""
Independent Diffusion Model for RSBench.

This model is identical to NeSyDM but enforces independence:
- Uses the same encoder (MNISTNeSyDiffEncoder)
- Uses MNISTIndepDiffClassifier instead of MNISTNeSyDiffClassifier

The key difference is that when predicting each concept, we only condition
on that concept's own diffusion state, not on other concepts' states.

This tests the hypothesis: Is cross-concept conditioning (joint modeling)
what prevents reasoning shortcuts, or is it something else?
"""

from typing_extensions import override

from expressive.args import RSBenchArguments
import torch
from torch import Tensor

from expressive.experiments.rsbench.datasets.utils.base_dataset import BaseDataset
from expressive.methods.base_model import BaseNeSyDiffusion, Problem
from expressive.methods.cond_model import CondNeSyDiffusion
from expressive.methods.simple_nesy_diff import SimpleNeSyDiffusion
from expressive.models.diffusion_model import WY_DATA, UnmaskingModel
import torch.nn as nn
from torch.nn.functional import one_hot


class RSBenchIndepDiffModel(UnmaskingModel):
    """
    Independent Diffusion Model - same as RSBenchModel but with independent classifier.

    When predicting p(c_i|x), only conditions on w_i (not w_j for j != i).
    """
    def __init__(self, args: RSBenchArguments, dataset: BaseDataset) -> None:
        super().__init__(
            vocab_dim=dataset.get_w_dim()[1],
            w_dims=dataset.get_w_dim()[0],
            seq_length=None,
            args=args,
        )
        # Use the independent backbone
        self.encoder, self.classifier = dataset.get_backbone_indep_diff()
        self.dataset_name = args.dataset
        self.dataset = dataset

    def encode_x(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def logits_t0(self, wy_t, x_encodings: Tensor, t: Tensor) -> Tensor:
        """
        Compute logits with INDEPENDENT predictions per concept.

        Each concept only sees its own diffusion state, not others.
        """
        w_SBW = wy_t
        if not isinstance(wy_t, Tensor):
            w_SBW = wy_t[0]
        one_hot_w = one_hot(w_SBW, self.vocab_dim + 1)

        # The independent classifier handles the factorization internally
        return self.classifier(x_encodings, one_hot_w)


class RSBenchIndepAdapter(Problem, nn.Module):
    """Same adapter as RSBench, works with independent diffusion."""
    def __init__(self, args: RSBenchArguments, dataset: BaseDataset):
        super().__init__()
        self.debug = args.DEBUG
        self.dataset = dataset

    @override
    def shape_w(self) -> torch.Size:
        return self.dataset.get_w_dim()

    @override
    def shape_y(self) -> torch.Size:
        return self.dataset.get_y_dim()

    @override
    def y_from_w(self, w_SBW: torch.Tensor) -> torch.Tensor:
        return self.dataset.y_from_w(w_SBW)


def create_indep_diffusion(args: RSBenchArguments, dataset: BaseDataset) -> BaseNeSyDiffusion:
    """Create an independent diffusion model."""
    model = RSBenchIndepDiffModel(args, dataset)
    problem = RSBenchIndepAdapter(args, dataset)
    if args.simple_model:
        return SimpleNeSyDiffusion(model, problem, args)
    else:
        return CondNeSyDiffusion(model, problem, args)
