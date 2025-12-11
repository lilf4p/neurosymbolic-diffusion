# MNIST Semantic Loss with TRULY Independent Encoders
#
# This implementation matches the Traffic Light example from the Semantic Loss paper:
# - Two SEPARATE neural networks with INDEPENDENT parameters (θ_1, θ_2)
# - Each network only sees its own input (x_1 or x_2)
# - p(c1, c2 | x) = p(c1 | x1; θ_1) * p(c2 | x2; θ_2)
#
# This is the TRUE independence assumption, unlike mnistsl which uses a shared encoder.

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
    parser = ArgumentParser(description="Learning via Concept Extractor with Independent Encoders.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class IndependentDigitEncoder(nn.Module):
    """
    A single encoder for ONE digit image.

    This matches the traffic light example where we have separate networks:
    - p_θr(red | x_r) for the red light
    - p_θg(green | x_g) for the green light

    Here we have:
    - p_θ1(digit | x_1) for the first digit
    - p_θ2(digit | x_2) for the second digit
    """

    def __init__(self, img_channels=1, hidden_channels=32, c_dim=10, dropout=0.5):
        super().__init__()

        self.c_dim = c_dim

        # Conv layers
        self.enc_block_1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_block_2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_block_3 = nn.Conv2d(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Output layer: predicts concept logits
        # For a 28x28 input: after 3 conv blocks with stride 2, we get 3x3 feature map
        # hidden_channels * 4 * 3 * 3 = 128 * 9 = 1152 (with hidden_channels=32)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_channels * 4 * 3 * 3, c_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass for a SINGLE digit image.

        Args:
            x: [B, 1, 28, 28] - single digit image

        Returns:
            logits: [B, c_dim] - concept logits for this digit
        """
        # Conv block 1
        x = self.enc_block_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Conv block 2
        x = self.enc_block_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Conv block 3
        x = self.enc_block_3(x)
        x = F.relu(x)

        # FC layer to concept logits
        x = self.flatten(x)
        logits = self.fc(x)

        return logits


class MnistSLIndep(CExt):
    """
    MNIST Semantic Loss with TRUE Independence.

    Uses TWO SEPARATE ENCODERS with INDEPENDENT PARAMETERS:
    - encoder_1 with parameters θ_1: processes digit 1 only
    - encoder_2 with parameters θ_2: processes digit 2 only

    This matches the traffic light example in the Semantic Loss paper exactly.

    Key difference from mnistsl:
    - mnistsl: uses ONE shared encoder for both digits (weight sharing)
    - mnistsl_indep: uses TWO separate encoders (no weight sharing)
    """

    NAME = "mnistsl_indep"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        # Don't use the passed encoder - we create our own independent encoders
        super(MnistSLIndep, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )

        self.args = args

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

        # ====================================================================
        # KEY DIFFERENCE: TWO INDEPENDENT ENCODERS
        # ====================================================================
        # Create TWO SEPARATE encoders with INDEPENDENT parameters
        # This matches: p(c1|x1; θ_1) and p(c2|x2; θ_2)
        self.encoder_1 = IndependentDigitEncoder(
            img_channels=1,
            hidden_channels=32,
            c_dim=self.n_facts,
            dropout=0.5
        )
        self.encoder_2 = IndependentDigitEncoder(
            img_channels=1,
            hidden_channels=32,
            c_dim=self.n_facts,
            dropout=0.5
        )
        # ====================================================================

        # Label predictor (for evaluation, not used in semantic loss)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_facts * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.nr_classes),
        )

        # ====================================================================
        # CONCEPT CONSISTENCY: Like denoising in NeSy Diffusion
        # ====================================================================
        # Weight for concept consistency loss (default 0 = disabled)
        self.w_consistency = getattr(args, 'w_consistency', 0.0)
        # Noise level for input augmentation (like diffusion noise)
        self.consistency_noise = getattr(args, 'consistency_noise', 0.3)
        # ====================================================================

        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)

    def add_noise_to_input(self, x, noise_level):
        """
        Add Gaussian noise to input images (like diffusion forward process).

        This is analogous to the q(w_t | w_0) step in NeSy diffusion,
        but applied to the input space instead of the concept space.
        """
        noise = torch.randn_like(x) * noise_level
        return x + noise

    def forward(self, x):
        """
        Forward pass with INDEPENDENT encoders.

        Each encoder only sees its own digit image.
        p(c1, c2 | x) = p(c1 | x1; θ_1) * p(c2 | x2; θ_2)

        When training with concept consistency (w_consistency > 0):
        - Also computes predictions on noisy inputs
        - Adds KL divergence loss to encourage consistent predictions
        """
        # Split input into two digit images
        # x: [B, 1, 28, 56] (two 28x28 images concatenated)
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        x1, x2 = xs[0], xs[1]  # Each is [B, 1, 28, 28]

        # ====================================================================
        # KEY: Each encoder ONLY sees its own input
        # ====================================================================
        logits_1 = self.encoder_1(x1)  # [B, n_facts] - p(c1 | x1; θ_1)
        logits_2 = self.encoder_2(x2)  # [B, n_facts] - p(c2 | x2; θ_2)
        # ====================================================================

        # Stack logits: [B, 2, n_facts]
        cs = torch.stack([logits_1, logits_2], dim=1)

        # Convert to probabilities
        pCs = self.normalize_concepts(cs)

        # Label prediction (concatenate logits)
        pred = self.mlp(cs.view(-1, self.n_facts * 2))

        # ====================================================================
        # CONCEPT CONSISTENCY LOSS (like w_denoising in NeSy Diffusion)
        # ====================================================================
        # During training: add noise to inputs and ensure consistent predictions
        # This is analogous to the denoising objective in diffusion models
        # ====================================================================
        if self.training and self.w_consistency > 0:
            with torch.no_grad():
                # Sample a random noise level (like random timestep in diffusion)
                t = torch.rand(1, device=x.device).item() * self.consistency_noise

            # Add noise to inputs
            x1_noisy = self.add_noise_to_input(x1, t)
            x2_noisy = self.add_noise_to_input(x2, t)

            # Get predictions on noisy inputs
            logits_1_noisy = self.encoder_1(x1_noisy)
            logits_2_noisy = self.encoder_2(x2_noisy)

            # Convert to probabilities
            prob_1 = F.softmax(logits_1, dim=-1)
            prob_2 = F.softmax(logits_2, dim=-1)
            prob_1_noisy = F.softmax(logits_1_noisy, dim=-1)
            prob_2_noisy = F.softmax(logits_2_noisy, dim=-1)

            # KL divergence: D_KL(p_clean || p_noisy)
            # We want noisy predictions to match clean predictions
            # KL(P||Q) = sum P * log(P/Q)
            eps = 1e-8
            kl_1 = (prob_1.detach() * (torch.log(prob_1.detach() + eps) - torch.log(prob_1_noisy + eps))).sum(dim=-1)
            kl_2 = (prob_2.detach() * (torch.log(prob_2.detach() + eps) - torch.log(prob_2_noisy + eps))).sum(dim=-1)

            # Store consistency loss (will be added by semantic loss)
            consistency_loss = (kl_1.mean() + kl_2.mean()) / 2
        else:
            consistency_loss = None

        return {"CS": cs, "YS": pred, "pCS": pCs, "CONSISTENCY_LOSS": consistency_loss}

    def normalize_concepts(self, z):
        """
        Computes the probability for each concept.

        Uses softmax to ensure probabilities sum to 1 for each digit.
        """
        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        # Softmax to get proper probabilities
        prob_digit1 = F.softmax(prob_digit1, dim=1)
        prob_digit2 = F.softmax(prob_digit2, dim=1)

        # Add small epsilon and renormalize for numerical stability
        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1

        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2

        return torch.stack([prob_digit1, prob_digit2], dim=1)

    def get_loss(self, args):
        """Returns the semantic loss function"""
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist", "permutedhalfmnist"]:
            return ADDMNIST_SL(ADDMNIST_Cumulative, self.logic, args)
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
