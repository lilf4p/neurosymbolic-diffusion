# MNIST Semantic Loss with One-Step Flow Denoising
#
# This implementation combines:
# 1. Independent encoders (like mnistsl_indep)
# 2. NeSy diffusion-style training (sample concepts → add noise → denoise)
# 3. One-step flow matching instead of multi-step diffusion
#
# The key insight from NeSy diffusion:
# - Sample concepts c ~ p_θ(c|x) from the encoder
# - Add noise to get c_t (in concept space, not input space!)
# - Train to reconstruct c_0 from c_t using flow matching
# - This provides a grounding signal that improves concept accuracy
#
# Flow matching advantage over diffusion:
# - Single-step denoising instead of iterative
# - Simpler training objective
# - No need for noise scheduling or multiple timesteps at inference

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
    parser = ArgumentParser(description="Semantic Loss with Independent Encoders and Flow Denoising.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class IndependentDigitEncoder(nn.Module):
    """
    A single encoder for ONE digit image.
    Same as in mnistsl_indep.
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

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_channels * 4 * 3 * 3, c_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass for a SINGLE digit image.
        """
        x = self.enc_block_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.enc_block_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.enc_block_3(x)
        x = F.relu(x)

        x = self.flatten(x)
        logits = self.fc(x)

        return logits


class ConceptFlowDenoiser(nn.Module):
    """
    One-step flow model for concept denoising.

    Takes noisy concept probabilities c_t and predicts clean concepts c_0.
    This is equivalent to predicting the "velocity" in flow matching,
    but since we want to go from c_t to c_0 in one step, we directly
    predict c_0.

    Architecture:
    - Input: noisy concept probs + timestep t + optional conditioning
    - Output: denoised concept logits
    """

    def __init__(self, c_dim=10, hidden_dim=128, n_concepts=2):
        super().__init__()

        self.c_dim = c_dim
        self.n_concepts = n_concepts

        # Input: noisy concepts (n_concepts * c_dim) + timestep (1)
        input_dim = n_concepts * c_dim + 1

        # Simple MLP for denoising
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_concepts * c_dim)
        )

    def forward(self, c_t, t):
        """
        Predict clean concepts from noisy concepts.

        Args:
            c_t: [B, n_concepts, c_dim] - noisy concept probabilities
            t: [B] or [B, 1] - timestep (noise level)

        Returns:
            c_0_pred: [B, n_concepts, c_dim] - predicted clean concept logits
        """
        B = c_t.shape[0]

        # Flatten concepts
        c_flat = c_t.view(B, -1)  # [B, n_concepts * c_dim]

        # Ensure t is the right shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]

        # Concatenate input
        x = torch.cat([c_flat, t], dim=-1)  # [B, n_concepts * c_dim + 1]

        # Predict clean concepts
        out = self.net(x)  # [B, n_concepts * c_dim]

        # Reshape back
        c_0_pred = out.view(B, self.n_concepts, self.c_dim)

        return c_0_pred


class MnistSLFlow(CExt):
    """
    MNIST Semantic Loss with One-Step Flow Denoising.

    Training procedure (like NeSy diffusion):
    1. Encoder predicts p(c|x) for each concept
    2. Sample concepts c_0 ~ p(c|x) using Gumbel-Softmax
    3. Add noise: c_t = (1-t) * c_0 + t * noise (flow interpolation)
    4. Train flow model to predict c_0 from c_t
    5. Use semantic loss on p(c|x) for label supervision

    The flow denoising provides the same grounding signal as
    L_w_denoising in NeSy diffusion, but in one step.
    """

    NAME = "mnistsl_flow"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        super(MnistSLFlow, self).__init__(
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
        # TWO INDEPENDENT ENCODERS (same as mnistsl_indep)
        # ====================================================================
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
        # ONE-STEP FLOW DENOISER
        # ====================================================================
        self.flow_denoiser = ConceptFlowDenoiser(
            c_dim=self.n_facts,
            hidden_dim=getattr(args, 'flow_hidden_dim', 128),
            n_concepts=2
        )

        # Flow denoising weight (like w_denoise_weight in NeSy diffusion)
        self.w_flow = getattr(args, 'w_flow', 1.0)

        # Temperature for Gumbel-Softmax sampling
        self.gumbel_tau = getattr(args, 'gumbel_tau', 1.0)

        # Whether to use hard (one-hot) or soft samples
        self.gumbel_hard = getattr(args, 'gumbel_hard', False)
        # ====================================================================

        # Label predictor
        self.mlp = nn.Sequential(
            nn.Linear(self.n_facts * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.nr_classes),
        )

        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)

    def sample_concepts_gumbel(self, logits, tau=1.0, hard=False):
        """
        Sample concepts using Gumbel-Softmax (differentiable sampling).

        Args:
            logits: [B, n_concepts, c_dim] - concept logits
            tau: temperature for Gumbel-Softmax
            hard: if True, return one-hot (straight-through gradient)

        Returns:
            samples: [B, n_concepts, c_dim] - sampled concept probabilities
        """
        # Gumbel-Softmax for differentiable sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = (logits + gumbel_noise) / tau
        y_soft = F.softmax(y, dim=-1)

        if hard:
            # Straight-through: forward uses argmax, backward uses softmax
            y_hard = F.one_hot(y_soft.argmax(dim=-1), logits.shape[-1]).float()
            return y_hard - y_soft.detach() + y_soft

        return y_soft

    def add_concept_noise(self, c_0, t):
        """
        Add noise to concepts using flow interpolation.

        c_t = (1 - t) * c_0 + t * noise

        where noise is uniform over the simplex (Dirichlet with alpha=1).

        Args:
            c_0: [B, n_concepts, c_dim] - clean concept samples
            t: [B] - timestep (noise level, 0 = clean, 1 = pure noise)

        Returns:
            c_t: [B, n_concepts, c_dim] - noisy concepts
        """
        # Sample noise from uniform simplex (Dirichlet with alpha=1)
        # This gives uniform distribution over probability simplices
        noise = torch.distributions.Dirichlet(
            torch.ones_like(c_0)
        ).sample()

        # Flow interpolation
        # Expand t to match dimensions
        t_expanded = t.view(-1, 1, 1)  # [B, 1, 1]

        c_t = (1 - t_expanded) * c_0 + t_expanded * noise

        return c_t

    def flow_denoising_loss(self, c_0, c_t, t, c_0_pred):
        """
        Flow denoising loss: MSE between predicted and true clean concepts.

        This is equivalent to L_w_denoising in NeSy diffusion:
        - We train the model to reconstruct the sampled concepts from noisy versions

        Args:
            c_0: [B, n_concepts, c_dim] - true clean concepts (sampled)
            c_t: [B, n_concepts, c_dim] - noisy concepts
            t: [B] - timestep
            c_0_pred: [B, n_concepts, c_dim] - predicted clean concept logits

        Returns:
            loss: scalar - denoising loss
        """
        # Convert predictions to probabilities
        c_0_pred_prob = F.softmax(c_0_pred, dim=-1)

        # MSE loss between predicted and true
        # (could also use cross-entropy or KL divergence)
        loss = F.mse_loss(c_0_pred_prob, c_0.detach())

        return loss

    def forward(self, x):
        """
        Forward pass with flow denoising.

        1. Encode: get p(c|x) from independent encoders
        2. Sample: c_0 ~ p(c|x) using Gumbel-Softmax
        3. Noise: c_t = interpolate(c_0, noise, t)
        4. Denoise: predict c_0 from c_t using flow model
        5. Loss: MSE(c_0_pred, c_0)
        """
        # Split input into two digit images
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        x1, x2 = xs[0], xs[1]

        # ====================================================================
        # ENCODING: Independent encoders
        # ====================================================================
        logits_1 = self.encoder_1(x1)  # [B, n_facts]
        logits_2 = self.encoder_2(x2)  # [B, n_facts]

        # Stack logits: [B, 2, n_facts]
        cs = torch.stack([logits_1, logits_2], dim=1)

        # Convert to probabilities
        pCs = self.normalize_concepts(cs)

        # Label prediction
        pred = self.mlp(cs.view(-1, self.n_facts * 2))

        # ====================================================================
        # FLOW DENOISING (during training only)
        # ====================================================================
        flow_loss = None
        if self.training and self.w_flow > 0:
            B = x.shape[0]

            # 1. Sample concepts from encoder distribution using Gumbel-Softmax
            c_0 = self.sample_concepts_gumbel(cs, tau=self.gumbel_tau, hard=self.gumbel_hard)

            # 2. Sample random timestep t ~ U(0, 1)
            t = torch.rand(B, device=x.device)

            # 3. Add noise to get c_t
            c_t = self.add_concept_noise(c_0, t)

            # 4. Predict clean concepts from noisy
            c_0_pred = self.flow_denoiser(c_t, t)

            # 5. Compute denoising loss
            flow_loss = self.flow_denoising_loss(c_0, c_t, t, c_0_pred)

        return {"CS": cs, "YS": pred, "pCS": pCs, "FLOW_LOSS": flow_loss}

    def normalize_concepts(self, z):
        """
        Computes the probability for each concept.
        """
        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        prob_digit1 = F.softmax(prob_digit1, dim=1)
        prob_digit2 = F.softmax(prob_digit2, dim=1)

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
        """Returns the semantic loss function with flow denoising"""
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist", "permutedhalfmnist"]:
            return ADDMNIST_SL_Flow(ADDMNIST_Cumulative, self.logic, args, self.w_flow)
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


class ADDMNIST_SL_Flow(ADDMNIST_SL):
    """
    Semantic loss with flow denoising loss integrated.

    Total loss = CE_label + w_sl * SL - w_h * H(c|x) + w_flow * L_flow
    """

    def __init__(self, cumulative_fn, logic, args, w_flow=1.0):
        super().__init__(cumulative_fn, logic, args)
        self.w_flow = w_flow

    def forward(self, out_dict, args):
        """Forward step with flow denoising loss added."""
        # Get base losses from parent
        loss, losses = super().forward(out_dict, args)

        # Add flow denoising loss
        flow_loss = out_dict.get("FLOW_LOSS")
        if flow_loss is not None:
            loss = loss + self.w_flow * flow_loss
            losses["flow_loss"] = flow_loss.item() if isinstance(flow_loss, torch.Tensor) else flow_loss

        return loss, losses