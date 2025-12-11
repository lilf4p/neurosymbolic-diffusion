# MNIST Semantic Loss with Noisy Concepts
#
# Key insight from diffusion experiments: noise injection during training
# prevents the model from collapsing to shortcuts.
#
# This model adds noise to concept probabilities during training,
# forcing the model to learn robust concept representations.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix
from utils.semantic_loss import ADDMNIST_SL


def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture"""
    parser = ArgumentParser(description="Learning via Concept Extractor with Noisy Training.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class NoisySLLoss(nn.Module):
    """
    Semantic Loss with noise injection during training.

    During training, we add noise to concept probabilities before computing
    the semantic loss. This prevents the model from finding shortcuts.
    """

    def __init__(self, base_loss, logic, args):
        super().__init__()
        self.base_loss = base_loss
        self.logic = logic
        self.args = args

        # Noise parameters
        self.noise_scale = getattr(args, 'noise_scale', 0.3)  # How much noise to add
        self.noise_type = getattr(args, 'noise_type', 'gaussian')  # gaussian or uniform

    def forward(self, out_dict, args):
        if self.training:
            # Add noise to concept probabilities during training
            pCs = out_dict["pCS"]  # [B, 2, n_facts]

            if self.noise_type == 'gaussian':
                # Add Gaussian noise in logit space
                logits = torch.log(pCs + 1e-8)
                noise = torch.randn_like(logits) * self.noise_scale
                noisy_logits = logits + noise
                noisy_pCs = F.softmax(noisy_logits, dim=-1)
            elif self.noise_type == 'uniform':
                # Uniform noise: mix with uniform distribution
                uniform = torch.ones_like(pCs) / pCs.shape[-1]
                alpha = torch.rand(pCs.shape[0], 1, 1, device=pCs.device) * self.noise_scale
                noisy_pCs = (1 - alpha) * pCs + alpha * uniform
            elif self.noise_type == 'dropout':
                # Random dropout of concept probabilities
                mask = torch.rand_like(pCs) > self.noise_scale
                noisy_pCs = pCs * mask.float()
                noisy_pCs = noisy_pCs / (noisy_pCs.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                noisy_pCs = pCs

            # Re-normalize
            noisy_pCs = noisy_pCs / (noisy_pCs.sum(dim=-1, keepdim=True) + 1e-8)

            # Create modified output dict
            out_dict_noisy = out_dict.copy()
            out_dict_noisy["pCS"] = noisy_pCs

            return self.base_loss(out_dict_noisy, args)
        else:
            # No noise during evaluation
            return self.base_loss(out_dict, args)


class MnistSLNoisy(CExt):
    """MNIST architecture with Noisy Semantic Loss Training"""

    NAME = "mnistsl_noisy"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        super(MnistSLNoisy, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )

        # Task-specific setup
        if args.task == "addition":
            self.n_facts = 10 if args.dataset not in ["halfmnist", "restrictedmnist", "permutedhalfmnist"] else 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "addmnist")
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = 10 if args.dataset not in ["halfmnist", "restrictedmnist", "permutedhalfmnist"] else 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "productmnist")
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "multiopmnist")
            self.nr_classes = 3

        self.args = args

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_facts * 2, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.nr_classes),
        )

        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)

    def forward(self, x):
        """Forward method"""
        # Image encoding
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _, _ = self.encoder(xs[i])
            cs.append(lc)
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        pCs = self.normalize_concepts(cs)

        pred = self.mlp(cs.view(-1, self.n_facts * 2))

        return {"CS": cs, "YS": pred, "pCS": pCs}

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        prob_digit1 = torch.nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = torch.nn.Softmax(dim=1)(prob_digit2)

        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1
        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2

        return torch.stack([prob_digit1, prob_digit2], dim=1).view(-1, 2, self.n_facts)

    def get_loss(self, args):
        """Returns the loss function with noisy training"""
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist", "permutedhalfmnist"]:
            base_loss = ADDMNIST_SL(ADDMNIST_Cumulative, self.logic, args)
            return NoisySLLoss(base_loss, self.logic, args)
        else:
            raise NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initializes the optimizer"""
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )

    def to(self, device):
        super().to(device)
        self.logic = self.logic.to(device)
