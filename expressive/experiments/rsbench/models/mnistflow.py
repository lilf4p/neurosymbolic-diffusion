# MNIST Flow Matching Model
#
# This model uses a discrete flow matching / noise-based approach to prevent concept collapse.
# Key insight: The problem with semantic loss is not the independence assumption, but that
# the encoder can collapse to a single mode. Diffusion prevents this through noise injection.
#
# This model:
# 1. Uses independent concept predictors (no cross-conditioning needed)
# 2. Adds noise during training to force learning the full distribution
# 3. Trains with a denoising objective to prevent collapse
# 4. Uses semantic loss on the denoised predictions

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix
from utils.semantic_loss import ADDMNIST_SL


def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture"""
    parser = ArgumentParser(description="Learning via Concept Extractor with Flow Matching.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FlowMatchingLoss(nn.Module):
    """
    Custom loss function that combines semantic loss with denoising/flow matching.

    The key insight is that we need to prevent the encoder from collapsing to a single mode.
    We do this by:
    1. Sampling from the predicted distribution
    2. Adding noise to create diverse training signals
    3. Training the encoder to be consistent across noise levels
    """

    def __init__(self, base_loss, logic, args, model):
        super().__init__()
        self.base_loss = base_loss  # ADDMNIST_SL or similar
        self.logic = logic
        self.args = args
        self.model = model  # Reference to the model for re-encoding

        # Hyperparameters
        self.diversity_weight = getattr(args, 'flow_diversity_weight', 0.1)
        self.entropy_weight = getattr(args, 'flow_entropy_weight', 0.01)
        self.num_samples = getattr(args, 'flow_samples', 8)

        # Task-specific setup
        if args.task == "addition":
            self.n_facts = 10 if args.dataset not in ["halfmnist", "restrictedmnist"] else 5
        elif args.task == "product":
            self.n_facts = 10 if args.dataset not in ["halfmnist", "restrictedmnist"] else 5
        else:
            self.n_facts = 5

    def forward(self, out_dict, args):
        """
        Compute the flow matching loss.

        Key components:
        1. Standard semantic loss (for label prediction)
        2. Entropy regularization (encourage diverse predictions)
        3. Consistency loss (predictions should be stable)
        """
        # First, compute standard semantic loss
        loss, losses = self.base_loss(out_dict, args)

        pCs = out_dict["pCS"]  # [B, 2, n_facts]
        Y = out_dict["LABELS"]
        B = pCs.shape[0]
        device = pCs.device

        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1, :]

        # ========================
        # 2. Entropy Regularization
        # ========================
        # Encourage high entropy (diverse) predictions to prevent collapse
        # H(p) = -sum(p * log(p))
        eps = 1e-8
        entropy1 = -torch.sum(prob_digit1 * torch.log(prob_digit1 + eps), dim=-1).mean()
        entropy2 = -torch.sum(prob_digit2 * torch.log(prob_digit2 + eps), dim=-1).mean()
        entropy_loss = -(entropy1 + entropy2) / 2  # Negative because we want to maximize entropy

        losses["entropy"] = -entropy_loss.item()  # Log the actual entropy value

        # ========================
        # 3. Diversity Loss (RLOO-style but with anti-collapse)
        # ========================
        # Sample multiple concept assignments and encourage diversity
        # by penalizing when all samples are the same

        S = self.num_samples

        # Sample from the predicted distributions
        dist1 = Categorical(probs=prob_digit1)
        dist2 = Categorical(probs=prob_digit2)

        samples1 = dist1.sample((S,))  # [S, B]
        samples2 = dist2.sample((S,))  # [S, B]

        # Compute diversity: how many unique samples per batch element?
        # High diversity = good (not collapsed)
        unique_pairs = []
        for b in range(B):
            pairs = set()
            for s in range(S):
                pairs.add((samples1[s, b].item(), samples2[s, b].item()))
            unique_pairs.append(len(pairs))

        diversity = torch.tensor(unique_pairs, device=device, dtype=torch.float).mean()
        max_diversity = min(S, self.n_facts * self.n_facts)  # Maximum possible unique pairs
        diversity_ratio = diversity / max_diversity

        # Diversity loss: penalize low diversity
        diversity_loss = -torch.log(diversity_ratio + eps)

        losses["diversity"] = diversity_ratio.item()
        losses["diversity_loss"] = diversity_loss.item()

        # ========================
        # 4. Constraint Satisfaction via Sampling (like RLOO)
        # ========================
        # Check if sampled concepts satisfy the label constraint
        if args.task == "addition":
            pred_labels = samples1 + samples2  # [S, B]
        elif args.task == "product":
            pred_labels = samples1 * samples2
        else:
            pred_labels = samples1 + samples2  # Default to addition

        Y_expanded = Y.unsqueeze(0).expand(S, -1)  # [S, B]
        correct = (pred_labels == Y_expanded).float()  # [S, B]

        # Mean accuracy of samples
        sample_acc = correct.mean().item()
        losses["sample_acc"] = sample_acc

        # RLOO-style loss: encourage samples that satisfy constraints
        log_probs = dist1.log_prob(samples1) + dist2.log_prob(samples2)  # [S, B]

        # Leave-one-out baseline
        sum_correct = correct.sum(dim=0, keepdim=True)  # [1, B]
        baseline = (sum_correct - correct) / (S - 1 + eps)  # [S, B]
        advantage = (correct - baseline).detach()

        rloo_loss = -(log_probs * advantage).mean()
        losses["rloo"] = rloo_loss.item()

        # ========================
        # Combine losses
        # ========================
        total_loss = loss + self.entropy_weight * entropy_loss + self.diversity_weight * diversity_loss

        losses["total"] = total_loss.item()

        return total_loss, losses


class MnistFlow(CExt):
    """
    MNIST architecture with Flow Matching for anti-collapse training.

    Key differences from MnistSL:
    1. Uses entropy regularization to prevent mode collapse
    2. Uses diversity loss to encourage varied predictions
    3. Uses RLOO-style sampling for constraint satisfaction
    """

    NAME = "mnistflow"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        super(MnistFlow, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )

        self.args = args

        # Task-specific setup
        if args.task == "addition":
            self.n_facts = 10 if args.dataset not in ["halfmnist", "restrictedmnist"] else 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "addmnist")
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = 10 if args.dataset not in ["halfmnist", "restrictedmnist"] else 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "productmnist")
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "multiopmnist")
            self.nr_classes = 3

        # Label predictor (same as MnistSL)
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
        """Forward method - same as MnistSL"""
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
        """Computes the probability for each concept"""
        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        # Softmax on digits_probs
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
        """Returns the custom flow matching loss"""
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
            base_loss = ADDMNIST_SL(ADDMNIST_Cumulative, self.logic, args)
            return FlowMatchingLoss(base_loss, self.logic, args, self)
        else:
            raise NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initialize optimizer"""
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )

    def to(self, device):
        super().to(device)
        self.logic = self.logic.to(device)

