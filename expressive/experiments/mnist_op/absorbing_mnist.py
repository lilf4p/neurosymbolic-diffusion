from typing_extensions import override
from typing import Union

import torch
from torch import Tensor
from torch.nn import Linear

from expressive.args import MNISTAbsorbingArguments
from expressive.experiments.mnist_op.models import MNISTEncoder, SimpleCNNNeSy
from expressive.methods.base_model import BaseNeSyDiffusion, Problem
from expressive.methods.cond_model import CondNeSyDiffusion
from expressive.methods.simple_nesy_diff import SimpleNeSyDiffusion
from expressive.models.diffusion_model import WY_DATA, UnmaskingModel
from expressive.models.dit import hidden_size


def vector_to_base10(w: torch.Tensor, N: int) -> torch.Tensor:
    device = w.device
    w = w.to(torch.int64)  # Convert to int64 for higher precision

    w_D = torch.zeros((*w.shape, N), dtype=torch.int64, device="cpu")
    if device.type == "mps":
        w = w.cpu()
    else:
        w = w

    for i in range(N):
        w_D[..., i] = (w // (10 ** (N - 1 - i))) % 10

    w_D = w_D.to(device)
    assert torch.all(torch.logical_and(0 <= w_D, w_D < 10))
    return w_D


class MNISTAbsorbModel(UnmaskingModel):
    def __init__(self, args: MNISTAbsorbingArguments) -> None:
        super().__init__(
            vocab_dim=10, w_dims=2 * args.N, seq_length=3 * args.N + 1, args=args
        )
        # Use embedding size from the DiT model
        print(hidden_size(args.model))
        self.encoder = MNISTEncoder(hidden_size(args.model))
        self.output_layer = Linear(hidden_size(args.model), 10)
        self.N = args.N

    def encode_x(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def logits_t0(self, wy_t: WY_DATA, x_encoding: Tensor, t: Tensor) -> Tensor:
        x_SBWE = torch.relu(x_encoding)
        return self.output_layer(x_SBWE)


class MNISTAddProblem(Problem):
    def __init__(self, args: MNISTAbsorbingArguments):
        self.N: int = args.N

    @override
    def shape_w(self) -> torch.Size:
        return (self.N * 2, 10)

    @override
    def shape_y(self) -> torch.Size:
        return (self.N + 1, 10)

    def y_from_w(self, w_SKB2xN: torch.Tensor) -> torch.Tensor:
        assert (w_SKB2xN < 10).all()  # Have to make sure no masked values are present
        stack1_SKBN = torch.stack(
            [10 ** (self.N - i - 1) * w_SKB2xN[..., i] for i in range(self.N)], -1
        )
        stack2_SKBN = torch.stack(
            [10 ** (self.N - i - 1) * w_SKB2xN[..., self.N + i] for i in range(self.N)],
            -1,
        )

        n1_SKB = stack1_SKBN.sum(-1)
        n2_SKB = stack2_SKBN.sum(-1)
        ty_SKB = n1_SKB + n2_SKB
        ty_SKBY = vector_to_base10(ty_SKB, self.N + 1)

        return ty_SKBY

def create_mnistadd(args: MNISTAbsorbingArguments) -> Union[BaseNeSyDiffusion, SimpleCNNNeSy]:
    """
    Factory function to create either a diffusion-based or CNN-based model
    for the MNIST addition task.

    Args:
        args: Configuration arguments containing model settings

    Returns:
        Either a diffusion model (BaseNeSyDiffusion) or a simple CNN model (SimpleCNNNeSy)
    """
    if args.use_cnn:
        print(f"Using SimpleCNNNeSy model with hidden_size={args.cnn_hidden_size}")
        return SimpleCNNNeSy(args)
    else:
        print(f"Using Diffusion model with hidden_size={hidden_size(args.model)}")
        model = MNISTAbsorbModel(args)
        problem = MNISTAddProblem(args)
        if args.simple_model:
            return SimpleNeSyDiffusion(model, problem, args)
        else:
            return CondNeSyDiffusion(model, problem, args)
