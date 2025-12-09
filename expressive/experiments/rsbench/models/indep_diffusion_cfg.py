"""
Independent Diffusion Model with Classifier-Free Guidance (CFG) for RSBench.

This model extends IndepDiffusion with CFG:
- During training: randomly drop the label with probability p_uncond
- During inference: interpolate between conditional and unconditional predictions

CFG provides stronger conditioning without a separate classifier.
"""

from expressive.args import RSBenchArguments
import torch
from torch import Tensor

from expressive.experiments.rsbench.datasets.utils.base_dataset import BaseDataset
from expressive.methods.base_model import BaseNeSyDiffusion, Problem
from expressive.methods.cfg_nesy_diff import CFGNeSyDiffusion
from expressive.models.diffusion_model import WY_DATA, UnmaskingModel
import torch.nn as nn
from torch.nn.functional import one_hot


class RSBenchIndepDiffModelCFG(UnmaskingModel):
    """
    Same as RSBenchIndepDiffModel but for CFG training.

    Uses the independent backbone where each concept only sees its own state.
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
        """
        w_SBW = wy_t
        if not isinstance(wy_t, Tensor):
            w_SBW = wy_t[0]
        one_hot_w = one_hot(w_SBW, self.vocab_dim + 1)

        return self.classifier(x_encodings, one_hot_w)


class RSBenchIndepAdapterCFG(Problem, nn.Module):
    """Same adapter as RSBench, works with CFG diffusion."""
    def __init__(self, args: RSBenchArguments, dataset: BaseDataset):
        super().__init__()
        self.debug = args.DEBUG
        self.dataset = dataset

    def shape_w(self) -> torch.Size:
        return self.dataset.get_w_dim()

    def shape_y(self) -> torch.Size:
        return self.dataset.get_y_dim()

    def y_from_w(self, w_SBW: torch.Tensor) -> torch.Tensor:
        return self.dataset.y_from_w(w_SBW)


def create_indep_diffusion_cfg(args: RSBenchArguments, dataset: BaseDataset) -> BaseNeSyDiffusion:
    """Create an independent diffusion model with classifier-free guidance."""
    model = RSBenchIndepDiffModelCFG(args, dataset)
    problem = RSBenchIndepAdapterCFG(args, dataset)

    return CFGNeSyDiffusion(model, problem, args)
