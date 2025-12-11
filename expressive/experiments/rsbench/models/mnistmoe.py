# MNIST Mixture of Experts Model
#
# This model uses multiple expert encoders to capture different modes of the
# concept distribution, preventing collapse to a single shortcut.
#
# Key idea: Instead of one encoder that collapses, have K encoders that are
# encouraged to specialize in different concept regions.

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
    parser = ArgumentParser(description="Learning via Concept Extractor with MoE.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class ExpertEncoder(nn.Module):
    """A single expert encoder for one concept."""
    def __init__(self, base_channels: int, c_dim: int):
        super().__init__()
        # Simple CNN for single digit
        self.conv1 = nn.Conv2d(1, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(base_channels * 2 * 7 * 7, c_dim)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MoELoss(nn.Module):
    """
    Loss function for Mixture of Experts model.

    Combines:
    1. Standard semantic loss
    2. Expert diversity loss (encourage different experts to specialize)
    """

    def __init__(self, base_loss, logic, args, model):
        super().__init__()
        self.base_loss = base_loss
        self.logic = logic
        self.args = args
        self.model = model

        # Hyperparameters
        self.diversity_weight = getattr(args, 'moe_diversity_weight', 0.1)

    def forward(self, out_dict, args):
        # Standard semantic loss
        loss, losses = self.base_loss(out_dict, args)

        # Get expert predictions for diversity computation
        expert_preds = out_dict.get("EXPERT_PROBS", None)  # [K, B, 2, c_dim]

        if expert_preds is not None and self.training:
            K = expert_preds.shape[0]
            B = expert_preds.shape[1]

            # Compute pairwise KL divergence between experts
            # High KL = different predictions = good diversity
            # Use cosine similarity between expert predictions instead of KL
            # Lower similarity = more diversity = what we want
            eps = 1e-6
            div_loss = 0
            count = 0
            for i in range(K):
                for j in range(i + 1, K):
                    # Flatten expert predictions
                    p_i = expert_preds[i].reshape(B, -1)  # [B, 2*c_dim]
                    p_j = expert_preds[j].reshape(B, -1)  # [B, 2*c_dim]

                    # Cosine similarity: high when experts agree, low when different
                    cos_sim = F.cosine_similarity(p_i, p_j, dim=-1).mean()

                    # We want LOW similarity (high diversity), so add similarity as loss
                    div_loss += cos_sim
                    count += 1

            if count > 0:
                div_loss = div_loss / count
                losses["diversity"] = div_loss.item()  # Log similarity (lower is better)
                loss = loss + self.diversity_weight * div_loss

        losses["total"] = loss.item()
        return loss, losses


class MnistMoE(CExt):
    """
    MNIST architecture with Mixture of Experts.

    Uses K expert encoders, each producing concept predictions.
    A gating network selects which expert's prediction to use.
    """

    NAME = "mnistmoe"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        super(MnistMoE, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )

        self.args = args
        self.num_experts = getattr(args, 'moe_num_experts', 4)

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

        # Keep the base encoder for feature extraction
        # But add expert heads on top
        self.expert_heads = nn.ModuleList([
            nn.Linear(self.n_facts, self.n_facts)  # Simple linear heads per expert
            for _ in range(self.num_experts)
        ])

        # Gating network: selects which expert to use based on input features
        self.gate = nn.Sequential(
            nn.Linear(self.n_facts * 2, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts),
        )

        # Label predictor
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
        """Forward method with MoE"""
        # Image encoding using base encoder
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _, _ = self.encoder(xs[i])
            cs.append(lc)
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)
        # cs: [B, 2, n_facts] - raw encoder outputs

        B = cs.shape[0]

        # Compute gating weights
        gate_input = cs.view(B, -1)  # [B, 2 * n_facts]
        gate_logits = self.gate(gate_input)  # [B, K]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, K]

        # Compute expert predictions
        expert_probs = []
        for k, head in enumerate(self.expert_heads):
            # Apply expert head to each concept
            expert_cs = torch.stack([
                head(cs[:, 0, :]),  # Expert k's prediction for digit 1
                head(cs[:, 1, :]),  # Expert k's prediction for digit 2
            ], dim=1)  # [B, 2, n_facts]
            expert_probs.append(F.softmax(expert_cs, dim=-1))

        expert_probs = torch.stack(expert_probs, dim=0)  # [K, B, 2, n_facts]

        # Weighted combination of expert predictions
        # gate_weights: [B, K] -> [K, B, 1, 1]
        gate_weights_expanded = gate_weights.T.unsqueeze(-1).unsqueeze(-1)  # [K, B, 1, 1]
        pCs = (gate_weights_expanded * expert_probs).sum(dim=0)  # [B, 2, n_facts]

        # Normalize
        pCs = self.normalize_concepts_simple(pCs)

        # Label prediction
        pred = self.mlp(cs.view(-1, self.n_facts * 2))

        return {
            "CS": cs,
            "YS": pred,
            "pCS": pCs,
            "EXPERT_PROBS": expert_probs,  # For diversity loss
            "GATE_WEIGHTS": gate_weights,
        }

    def normalize_concepts_simple(self, pCs):
        """Simple normalization"""
        eps = 1e-5
        pCs = pCs + eps
        with torch.no_grad():
            Z = pCs.sum(dim=-1, keepdim=True)
        return pCs / Z

    def get_loss(self, args):
        """Returns the MoE loss"""
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist", "permutedhalfmnist"]:
            base_loss = ADDMNIST_SL(ADDMNIST_Cumulative, self.logic, args)
            return MoELoss(base_loss, self.logic, args, self)
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
