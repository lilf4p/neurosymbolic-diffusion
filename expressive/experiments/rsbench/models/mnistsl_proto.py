# MNIST Semantic Loss with Prototype-based Concept Learning
#
# Key insight: The problem with semantic loss is that it only provides
# supervision at the output level (Y), not at the concept level (C).
# The model can satisfy the constraint with degenerate concept predictions.
#
# Solution: Add concept prototypes that act as "anchors" in the latent space.
# Without direct concept supervision, we use the constraint satisfaction
# signal to pull embeddings toward meaningful prototypes.
#
# Inspired by:
# - Prototypical Networks (Snell et al., 2017)
# - SwAV (Caron et al., 2020) - prototype-based self-supervised learning
# - NeSy Diffusion's concept unmasking objective

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
    parser = ArgumentParser(description="Learning via Concept Extractor with Prototypes.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class DigitEncoderWithEmbedding(nn.Module):
    """
    Encoder that outputs both concept logits AND an embedding vector.

    The logits go through a direct path (same as mnistsl_indep).
    The embedding is a separate branch for prototype matching.
    """

    def __init__(self, img_channels=1, hidden_channels=32, c_dim=10, embed_dim=64, dropout=0.5):
        super().__init__()

        self.c_dim = c_dim
        self.embed_dim = embed_dim

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
        feature_dim = hidden_channels * 4 * 3 * 3  # 1152 for hidden_channels=32

        # Direct path to logits (same as mnistsl_indep)
        self.logit_fc = nn.Linear(feature_dim, c_dim)

        # Separate embedding branch (for prototype matching only)
        self.embed_fc = nn.Linear(feature_dim, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass.

        Returns:
            embed: [B, embed_dim] - embedding for prototype matching
            logits: [B, c_dim] - concept logits (direct path)
        """
        # Conv blocks
        x = self.enc_block_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.enc_block_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.enc_block_3(x)
        x = F.relu(x)

        # Flatten
        features = self.flatten(x)

        # Direct path to logits (same as mnistsl_indep)
        logits = self.logit_fc(features)

        # Separate embedding branch
        embed = self.embed_fc(features)
        embed = F.relu(embed)
        embed_norm = F.normalize(embed, p=2, dim=-1)

        return embed_norm, logits
class MnistSLProto(CExt):
    """
    MNIST Semantic Loss with Prototype-based Concept Learning.

    Key components:
    1. Learned prototypes for each digit class
    2. Prototype assignment based on concept predictions
    3. Contrastive-style loss to pull embeddings toward assigned prototypes

    The intuition: If the semantic constraint is satisfied, the predicted
    concepts must be "close to correct". We use this signal to pull the
    embedding toward the prototype of the predicted class, which gradually
    refines both the prototypes and the encoder.
    """

    NAME = "mnistsl_proto"

    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        super(MnistSLProto, self).__init__(
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

        # Hyperparameters
        self.embed_dim = getattr(args, 'embed_dim', 64)
        self.w_proto = getattr(args, 'w_proto', 1.0)  # Prototype loss weight
        self.proto_temp = getattr(args, 'proto_temp', 0.1)  # Temperature for softmax
        self.proto_momentum = getattr(args, 'proto_momentum', 0.9)  # EMA momentum

        # Two independent encoders
        self.encoder_1 = DigitEncoderWithEmbedding(
            img_channels=1,
            hidden_channels=32,
            c_dim=self.n_facts,
            embed_dim=self.embed_dim,
            dropout=0.5
        )
        self.encoder_2 = DigitEncoderWithEmbedding(
            img_channels=1,
            hidden_channels=32,
            c_dim=self.n_facts,
            embed_dim=self.embed_dim,
            dropout=0.5
        )

        # ====================================================================
        # CONCEPT PROTOTYPES
        # ====================================================================
        # Learned prototypes for each digit class
        # These are L2-normalized and updated via gradient descent
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(self.n_facts, self.embed_dim), p=2, dim=-1)
        )

        # EMA prototypes (not learnable, updated via momentum)
        self.register_buffer(
            'ema_prototypes',
            F.normalize(torch.randn(self.n_facts, self.embed_dim), p=2, dim=-1)
        )
        # ====================================================================

        # Label predictor
        self.mlp = nn.Sequential(
            nn.Linear(self.n_facts * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.nr_classes),
        )

        # Pending embeddings for EMA update (set during forward, used after backward)
        self._pending_embeds = None
        self._pending_logits = None

        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)

    def compute_prototype_loss(self, embed, logits, use_ema=True):
        """
        Compute prototype-based loss.

        The idea: Use predicted concept probabilities to create a "soft"
        prototype target, then pull the embedding toward this target.

        This is like SwAV but using concept predictions instead of clustering.

        Args:
            embed: [B, embed_dim] - L2-normalized embeddings
            logits: [B, n_facts] - concept logits
            use_ema: Whether to use EMA prototypes (more stable)

        Returns:
            loss: Prototype alignment loss
        """
        # Get concept probabilities
        probs = F.softmax(logits, dim=-1)  # [B, n_facts]

        # Get prototypes (normalized)
        if use_ema:
            protos = self.ema_prototypes  # [n_facts, embed_dim]
        else:
            protos = F.normalize(self.prototypes, p=2, dim=-1)

        # Compute similarity between embeddings and all prototypes
        # embed: [B, embed_dim], protos: [n_facts, embed_dim]
        sim = torch.mm(embed, protos.t())  # [B, n_facts] - cosine similarity

        # Scale by temperature and apply softmax
        proto_probs = F.softmax(sim / self.proto_temp, dim=-1)  # [B, n_facts]

        # Cross-entropy between concept probs and prototype probs
        # This encourages: if you predict digit X, your embedding should be
        # close to prototype X
        eps = 1e-8
        loss = -(probs.detach() * torch.log(proto_probs + eps)).sum(dim=-1).mean()

        # Also add reverse direction: if embedding is close to prototype X,
        # you should predict digit X
        loss_rev = -(proto_probs.detach() * torch.log(probs + eps)).sum(dim=-1).mean()

        return (loss + loss_rev) / 2

    @torch.no_grad()
    def update_ema_prototypes(self, embeds, logits):
        """
        Update EMA prototypes based on current batch.

        For each class, compute the mean embedding of samples predicted
        as that class, then update the EMA prototype.
        """
        probs = F.softmax(logits, dim=-1)  # [B, n_facts]

        # Hard assignment (argmax)
        assignments = probs.argmax(dim=-1)  # [B]

        for c in range(self.n_facts):
            mask = (assignments == c)
            if mask.sum() > 0:
                # Mean embedding of samples assigned to class c
                class_embed = embeds[mask].mean(dim=0)
                class_embed = F.normalize(class_embed, p=2, dim=0)

                # EMA update
                self.ema_prototypes[c] = (
                    self.proto_momentum * self.ema_prototypes[c] +
                    (1 - self.proto_momentum) * class_embed
                )

        # Renormalize
        self.ema_prototypes.copy_(
            F.normalize(self.ema_prototypes, p=2, dim=-1)
        )

    def post_backward(self):
        """
        Call this after backward() to update EMA prototypes.
        This must be done after backward to avoid in-place modification errors.
        """
        if self._pending_embeds is not None and self._pending_logits is not None:
            embed_1, embed_2 = self._pending_embeds
            logits_1, logits_2 = self._pending_logits
            all_embeds = torch.cat([embed_1, embed_2], dim=0)
            all_logits = torch.cat([logits_1, logits_2], dim=0)
            self.update_ema_prototypes(all_embeds, all_logits)
            self._pending_embeds = None
            self._pending_logits = None

    def forward(self, x):
        """
        Forward pass with prototype-based concept learning.
        """
        # Split input into two digit images
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        x1, x2 = xs[0], xs[1]

        # Get embeddings and logits from each encoder
        embed_1, logits_1 = self.encoder_1(x1)
        embed_2, logits_2 = self.encoder_2(x2)

        # Stack logits: [B, 2, n_facts]
        cs = torch.stack([logits_1, logits_2], dim=1)

        # Convert to probabilities
        pCs = self.normalize_concepts(cs)

        # Label prediction
        pred = self.mlp(cs.view(-1, self.n_facts * 2))

        # Compute prototype loss during training
        if self.training and self.w_proto > 0:
            proto_loss_1 = self.compute_prototype_loss(embed_1, logits_1)
            proto_loss_2 = self.compute_prototype_loss(embed_2, logits_2)
            proto_loss = (proto_loss_1 + proto_loss_2) / 2

            # Store embeddings for EMA update AFTER backward
            # (will be called by training loop or we register a hook)
            self._pending_embeds = (embed_1.detach().clone(), embed_2.detach().clone())
            self._pending_logits = (logits_1.detach().clone(), logits_2.detach().clone())
        else:
            proto_loss = None

        return {
            "CS": cs,
            "YS": pred,
            "pCS": pCs,
            "PROTO_LOSS": proto_loss,
            "EMBEDS": (embed_1, embed_2)  # For analysis
        }

    def normalize_concepts(self, z):
        """Computes the probability for each concept."""
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
