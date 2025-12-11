# MNIST Semantic Loss with Adaptive Loss Weighting
#
# Key insight from NeSy diffusion:
# - Y accuracy increases slowly, giving time for C to learn properly
# - In standard semantic loss, Y reaches ~99% quickly, then C stops improving
#
# Solution: Adaptive loss weighting
# - Track Y accuracy during training
# - When Y accuracy is high, reduce Y loss weight
# - This forces the model to keep improving C even when Y is "good enough"
#
# Additional techniques:
# 1. Label smoothing - prevent Y from reaching 1.0
# 2. Y temperature - soften Y predictions to maintain gradients
# 3. Concept reconstruction - auxiliary loss for C grounding

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
    parser = ArgumentParser(description="Semantic Loss with Adaptive Weighting.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class IndependentDigitEncoder(nn.Module):
    """A single encoder for ONE digit image."""

    def __init__(self, img_channels=1, hidden_channels=32, c_dim=10, dropout=0.5):
        super().__init__()
        self.c_dim = c_dim

        self.enc_block_1 = nn.Conv2d(img_channels, hidden_channels, 4, 2, 1)
        self.enc_block_2 = nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1)
        self.enc_block_3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, 2, 1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_channels * 4 * 3 * 3, c_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.enc_block_1(x))
        x = self.dropout(x)
        x = F.relu(self.enc_block_2(x))
        x = self.dropout(x)
        x = F.relu(self.enc_block_3(x))
        return self.fc(self.flatten(x))


class ConceptReconstructor(nn.Module):
    """
    Auxiliary network for concept reconstruction.

    Given noisy/perturbed concept predictions, reconstruct the original.
    This provides a grounding signal similar to denoising in diffusion.
    """

    def __init__(self, c_dim=10, hidden_dim=64, n_concepts=2):
        super().__init__()
        self.c_dim = c_dim
        self.n_concepts = n_concepts

        input_dim = n_concepts * c_dim + 1  # concepts + noise level

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_concepts * c_dim)
        )

    def forward(self, c_noisy, noise_level):
        B = c_noisy.shape[0]
        c_flat = c_noisy.view(B, -1)

        if noise_level.dim() == 0:
            noise_level = noise_level.unsqueeze(0).expand(B)
        if noise_level.dim() == 1:
            noise_level = noise_level.unsqueeze(-1)

        x = torch.cat([c_flat, noise_level], dim=-1)
        out = self.net(x)
        return out.view(B, self.n_concepts, self.c_dim)


class MnistSLAdaptive(CExt):
    """
    MNIST Semantic Loss with Adaptive Loss Weighting.

    Features:
    1. Independent encoders (like mnistsl_indep)
    2. Adaptive Y loss weighting based on running accuracy
    3. Label smoothing to prevent Y saturation
    4. Concept reconstruction loss for grounding
    """

    NAME = "mnistsl_adaptive"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        super(MnistSLAdaptive, self).__init__(
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

        # Independent encoders
        self.encoder_1 = IndependentDigitEncoder(
            img_channels=1, hidden_channels=32, c_dim=self.n_facts, dropout=0.5
        )
        self.encoder_2 = IndependentDigitEncoder(
            img_channels=1, hidden_channels=32, c_dim=self.n_facts, dropout=0.5
        )

        # Label predictor
        self.mlp = nn.Sequential(
            nn.Linear(self.n_facts * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.nr_classes),
        )

        # ====================================================================
        # ADAPTIVE LOSS PARAMETERS
        # ====================================================================

        # Label smoothing factor (0 = no smoothing, 0.1 = 10% smoothing)
        self.label_smoothing = getattr(args, 'label_smoothing', 0.1)

        # Y temperature for softening predictions (higher = softer)
        self.y_temperature = getattr(args, 'y_temperature', 1.0)

        # Adaptive weighting parameters
        self.y_acc_threshold = getattr(args, 'y_acc_threshold', 0.95)  # When to start reducing Y weight
        self.y_weight_min = getattr(args, 'y_weight_min', 0.1)  # Minimum Y loss weight
        self.adaptive_y_weight = getattr(args, 'adaptive_y_weight', True)

        # Concept reconstruction weight
        self.w_recon = getattr(args, 'w_recon', 1.0)
        self.recon_noise = getattr(args, 'recon_noise', 0.3)

        # Running Y accuracy tracker (exponential moving average)
        self.register_buffer('running_y_acc', torch.tensor(0.5))
        self.ema_decay = 0.99

        # Concept reconstructor
        if self.w_recon > 0:
            self.reconstructor = ConceptReconstructor(
                c_dim=self.n_facts, hidden_dim=64, n_concepts=2
            )
        else:
            self.reconstructor = None

        # ====================================================================

        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)

    def compute_adaptive_y_weight(self):
        """
        Compute adaptive weight for Y loss based on running accuracy.

        When Y accuracy is above threshold, reduce the weight to prevent
        the model from "resting" on its laurels and ignoring concept learning.
        """
        if not self.adaptive_y_weight:
            return 1.0

        y_acc = self.running_y_acc.item()

        if y_acc < self.y_acc_threshold:
            # Below threshold: full weight
            return 1.0
        else:
            # Above threshold: linearly decay weight
            # At threshold: weight = 1.0
            # At 1.0: weight = y_weight_min
            excess = (y_acc - self.y_acc_threshold) / (1.0 - self.y_acc_threshold)
            weight = 1.0 - excess * (1.0 - self.y_weight_min)
            return max(self.y_weight_min, weight)

    def add_concept_noise(self, c_probs, noise_level):
        """Add noise to concept probabilities for reconstruction."""
        # Sample noise from Dirichlet (uniform over simplex)
        noise = torch.distributions.Dirichlet(torch.ones_like(c_probs)).sample()

        # Interpolate between clean and noise
        if noise_level.dim() == 0:
            noise_level = noise_level.unsqueeze(0)
        noise_level = noise_level.view(-1, 1, 1)

        c_noisy = (1 - noise_level) * c_probs + noise_level * noise
        return c_noisy

    def forward(self, x):
        # Split input
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        x1, x2 = xs[0], xs[1]

        # Independent encoding
        logits_1 = self.encoder_1(x1)
        logits_2 = self.encoder_2(x2)

        # Stack logits
        cs = torch.stack([logits_1, logits_2], dim=1)

        # Get probabilities
        pCs = self.normalize_concepts(cs)

        # Label prediction with optional temperature
        pred_logits = self.mlp(cs.view(-1, self.n_facts * 2))
        if self.y_temperature != 1.0:
            pred_logits = pred_logits / self.y_temperature

        # ====================================================================
        # CONCEPT RECONSTRUCTION LOSS
        # ====================================================================
        recon_loss = None
        if self.training and self.w_recon > 0 and self.reconstructor is not None:
            B = x.shape[0]

            # Sample noise level
            t = torch.rand(B, device=x.device) * self.recon_noise

            # Add noise to concept probabilities
            c_noisy = self.add_concept_noise(pCs.detach(), t)

            # Reconstruct clean concepts
            c_recon_logits = self.reconstructor(c_noisy, t)
            c_recon = F.softmax(c_recon_logits, dim=-1)

            # MSE loss between reconstructed and original
            recon_loss = F.mse_loss(c_recon, pCs.detach())

        return {
            "CS": cs,
            "YS": pred_logits,
            "pCS": pCs,
            "RECON_LOSS": recon_loss,
            "Y_WEIGHT": self.compute_adaptive_y_weight(),
            "RUNNING_Y_ACC": self.running_y_acc.item(),
        }

    def update_running_y_acc(self, y_pred, y_true):
        """Update running Y accuracy with EMA."""
        with torch.no_grad():
            acc = (y_pred.argmax(dim=-1) == y_true).float().mean()
            self.running_y_acc = self.ema_decay * self.running_y_acc + (1 - self.ema_decay) * acc

    def normalize_concepts(self, z):
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
        """Returns the adaptive semantic loss function"""
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist", "permutedhalfmnist"]:
            return ADDMNIST_SL_Adaptive(
                ADDMNIST_Cumulative, self.logic, args, self
            )
        else:
            raise NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )

    def to(self, device):
        super().to(device)
        self.logic = self.logic.to(device)


class ADDMNIST_SL_Adaptive(ADDMNIST_SL):
    """
    Semantic loss with adaptive weighting.

    Features:
    1. Label smoothing
    2. Adaptive Y loss weight based on running accuracy
    3. Concept reconstruction loss
    """

    def __init__(self, cumulative_fn, logic, args, model):
        super().__init__(cumulative_fn, logic, args)
        self.model = model
        self.label_smoothing = getattr(args, 'label_smoothing', 0.1)
        self.w_recon = getattr(args, 'w_recon', 1.0)

    def forward(self, out_dict, args):
        """
        Compute adaptive loss.

        Note: Uses forward() to match parent class signature.
        """
        Y = out_dict["LABELS"]
        pCs = out_dict["pCS"]
        Y_pred = out_dict["YS"]

        # Update running Y accuracy
        if self.model.training:
            self.model.update_running_y_acc(Y_pred, Y)

        # ================================================================
        # LABEL LOSS WITH SMOOTHING AND ADAPTIVE WEIGHTING
        # ================================================================
        if self.label_smoothing > 0:
            # Label smoothing
            n_classes = Y_pred.shape[-1]
            smooth_labels = torch.full_like(Y_pred, self.label_smoothing / n_classes)
            smooth_labels.scatter_(1, Y.unsqueeze(1).long(), 1 - self.label_smoothing + self.label_smoothing / n_classes)
            y_loss = F.cross_entropy(Y_pred, smooth_labels, reduction="mean")
        else:
            y_loss = F.cross_entropy(Y_pred, Y.long(), reduction="mean")

        # Apply adaptive weight
        y_weight = out_dict.get("Y_WEIGHT", 1.0)
        y_loss_weighted = y_weight * y_loss

        losses = {
            "y-loss": y_loss.item(),
            "y-weight": y_weight,
            "running_y_acc": out_dict.get("RUNNING_Y_ACC", 0.0),
        }

        # ================================================================
        # SEMANTIC LOSS
        # ================================================================
        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1, :]

        Z_1 = prob_digit1[..., None]
        Z_2 = prob_digit2[:, None, :]
        probs = Z_1.multiply(Z_2)
        worlds_prob = probs.reshape(-1, self.n_facts * self.n_facts)

        query_prob = torch.zeros(
            size=(len(probs), self.nr_classes), device=probs.device
        )
        for i in range(self.nr_classes):
            query_prob[:, i] = self.compute_query(i, worlds_prob).view(-1)

        eps = 1e-5
        query_prob = query_prob + eps
        with torch.no_grad():
            Z = torch.sum(query_prob, dim=-1, keepdim=True)
        query_prob = query_prob / Z

        sl = F.nll_loss(query_prob.log(), Y.to(torch.long), reduction="mean")
        losses["sl"] = sl.item()

        # ================================================================
        # ENTROPY REGULARIZATION
        # ================================================================
        if self.w_h > 0:
            eps_h = 1e-8
            entropy_1 = -torch.sum(prob_digit1 * torch.log(prob_digit1 + eps_h), dim=-1)
            entropy_2 = -torch.sum(prob_digit2 * torch.log(prob_digit2 + eps_h), dim=-1)
            entropy = (entropy_1 + entropy_2).mean() / 2
            losses["entropy"] = entropy.item()
        else:
            entropy = 0.0

        # ================================================================
        # CONCEPT RECONSTRUCTION LOSS
        # ================================================================
        recon_loss = out_dict.get("RECON_LOSS", None)
        if recon_loss is not None and self.w_recon > 0:
            losses["recon_loss"] = recon_loss.item()
            recon_term = self.w_recon * recon_loss
        else:
            recon_term = 0.0

        # ================================================================
        # TOTAL LOSS
        # ================================================================
        # Note: When Y accuracy is high, y_weight decreases, making the model
        # focus more on semantic loss and reconstruction
        total_loss = y_loss_weighted + args.w_sl * sl - self.w_h * entropy + recon_term

        return total_loss, losses
