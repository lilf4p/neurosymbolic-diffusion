# mnist sl module with suggested architecture
import torch
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix
from utils.semantic_loss import ADDMNIST_SL


def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture"""
    parser = ArgumentParser(description="Learning via Concept Extractor with Suggested Architecture.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistSLSuggested(CExt):
    """MNIST architecture with SL using the suggested encoder architecture.

    This follows the recommendations:
    - Conv(5×5, 32), ReLU, MaxPool(2)
    - Conv(5×5, 64), ReLU, MaxPool(2)
    - Flatten → FC 128, ReLU
    - Head per concept: FC 64 → ReLU → C-way softmax
    """

    NAME = "mnistslsuggested"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        """Initialize method

        Args:
            encoder (nn.Module): encoder network (will be replaced with suggested one)
            n_images (int, default=2): number of images
            c_split: concept split
            args: command line arguments
            n_facts (int, default=20): number of concepts
            nr_classes (int, default=19): number of classes
        """
        super(MnistSLSuggested, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )

        # Worlds-queries matrix
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

        # Use the suggested larger MLP for label prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_facts * 2, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.nr_classes),
        )

        # opt and device
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

        # Label prediction
        pred = self.mlp(cs.view(-1, self.n_facts * 2))

        return {"CS": cs, "YS": pred, "pCS": pCs}

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each concept using softmax"""
        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        # Softmax on digits_probs (mutually exclusive)
        prob_digit1 = torch.nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = torch.nn.Softmax(dim=1)(prob_digit2)

        # Clamp to avoid underflow
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
        """Returns the loss function"""
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
            return ADDMNIST_SL(ADDMNIST_Cumulative, self.logic, args)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initializes the optimizer"""
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )

    def to(self, device):
        super().to(device)
        self.logic = self.logic.to(device)
        return self
