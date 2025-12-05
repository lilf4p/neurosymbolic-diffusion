# Independent Factorized Model for MNIST
# This model uses the same encoder as NeSyDM but assumes independence:
# p(c|x) = prod_i p(c_i|x) instead of modeling the joint distribution
import torch
import torch.nn as nn
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix
from utils.semantic_loss import ADDMNIST_SL


def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor with Independence Assumption.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistIndep(CExt):
    """MNIST architecture with Independence Assumption + Semantic Loss

    This model is designed to test whether the independence assumption
    p(c|x) = prod_i p(c_i|x) causes reasoning shortcuts, or if the issue
    lies elsewhere (as suggested by recent research).

    Key differences from NeSyDM:
    - Uses factorized conditional probabilities instead of joint diffusion
    - Same encoder backbone structure
    - Uses semantic loss for constraint satisfaction
    """

    NAME = "mnistindep"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        """Initialize method

        Args:
            encoder (nn.Module): encoder network
            n_images (int, default=2): number of images
            c_split: concept split
            args: command line arguments
            n_facts (int, default=20): number of concepts
            nr_classes (int, default=19): number of classes
        """
        super(MnistIndep, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )

        # Setup based on task
        if args.task == "addition":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "addmnist")
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "productmnist")
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "multiopmnist")
            self.nr_classes = 3

        # MLP for label prediction (similar to mnistsl)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_facts * 2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, self.nr_classes),
        )

        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)

    def forward(self, x):
        """Forward method with factorized concept predictions

        Args:
            x (torch.tensor): input images concatenated

        Returns:
            out_dict: model predictions with:
                - CS: raw concept scores
                - YS: label predictions
                - pCS: normalized concept probabilities (factorized)
        """
        # Image encoding - same as mnistsl
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _, _ = self.encoder(xs[i])  # lc is concept logits
            cs.append(lc)
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        # Normalize concepts to probabilities (factorized - independent softmax)
        pCs = self.normalize_concepts(cs)

        # Compute label predictions via MLP
        pred = self.mlp(cs.view(-1, self.n_facts * 2))

        return {"CS": cs, "YS": pred, "pCS": pCs}

    def normalize_concepts(self, z, split=2):
        """Normalize concept predictions to probabilities.

        Key point: This uses INDEPENDENT softmax for each concept,
        implementing the factorization p(c|x) = prod_i p(c_i|x)

        Args:
            z (torch.tensor): raw concept logits [batch, n_images, n_facts]
            split (int, default=2): number of splits (images)

        Returns:
            pCs: normalized probabilities [batch, n_images, n_facts]
        """
        # Extract probs for each digit
        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        # Softmax on digits_probs (independent assumption: each digit's values are mutually exclusive)
        prob_digit1 = torch.nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = torch.nn.Softmax(dim=1)(prob_digit2)

        # Clamp digits_probs to avoid underflow
        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1  # Normalization
        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2  # Normalization

        return torch.stack([prob_digit1, prob_digit2], dim=1).view(-1, 2, self.n_facts)

    def get_loss(self, args):
        """Returns the loss function (Semantic Loss with independence assumption)

        Args:
            args: command line arguments

        Returns:
            loss: semantic loss function
        """
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
            return ADDMNIST_SL(ADDMNIST_Cumulative, self.logic, args)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initializes the optimizer

        Args:
            args: command line arguments
        """
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )

    def to(self, device):
        """Move model to device"""
        super().to(device)
        self.logic = self.logic.to(device)
        return self
