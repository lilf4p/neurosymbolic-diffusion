"""

NeSy CNN model with Semantic Loss, Entropy Regularization, and Topological Consistency.

Key components:
1. Semantic Loss: -log p(y|x) where p(y|x) = sum_w p(w|x) * 1{f(w)=y}
2. Entropy Regularization: Maximize H(w|x) or H(w|x,y) to prevent shortcuts
3. Topological Consistency: Enforce latent geometry matches concept geometry (NEW)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


@dataclass
class IterNeSyLog:
    """Training log."""
    accuracy_w: float = 0.0
    accuracy_y: float = 0.0
    entropy: float = 0.0
    cond_entropy: float = 0.0
    uncond_entropy: float = 0.0
    topo_loss: float = 0.0
    loss: float = 0.0
    n_batches: int = 0

    def reset(self):
        self.accuracy_w = 0.0
        self.accuracy_y = 0.0
        self.entropy = 0.0
        self.cond_entropy = 0.0
        self.uncond_entropy = 0.0
        self.topo_loss = 0.0
        self.loss = 0.0
        self.n_batches = 0


class TopologicalConsistencyLoss(nn.Module):
    """
    Enforces that the geometry of the latent/embedding space matches
    the geometry of the concept space.

    This provides an inductive bias: if concepts are "close" (e.g., (0,1) vs (0,2)),
    their learned representations should also be close. This helps OOD generalization
    by ensuring the model learns a geometrically meaningful representation.

    Multiple distance metrics and normalization strategies available.
    """

    def __init__(
        self,
        distance_metric: str = 'l2',
        normalization: str = 'mean',
        use_soft_rank: bool = False,
    ):
        """
        Args:
            distance_metric: 'l2', 'l1', or 'cosine'
            normalization: 'mean' (divide by mean), 'minmax' (scale to [0,1]), or 'none'
            use_soft_rank: If True, compare rank orderings instead of absolute distances
        """
        super().__init__()
        self.distance_metric = distance_metric
        self.normalization = normalization
        self.use_soft_rank = use_soft_rank

    def compute_pairwise_distance(self, x: Tensor, metric: str = 'l2') -> Tensor:
        """Compute pairwise distance matrix."""
        if metric == 'l2':
            return torch.cdist(x, x, p=2)
        elif metric == 'l1':
            return torch.cdist(x, x, p=1)
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            x_norm = F.normalize(x, p=2, dim=-1)
            sim = torch.mm(x_norm, x_norm.t())
            return 1 - sim
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def normalize_distances(self, dist: Tensor) -> Tensor:
        """Normalize distance matrix."""
        if self.normalization == 'mean':
            return dist / (dist.mean() + 1e-8)
        elif self.normalization == 'minmax':
            d_min, d_max = dist.min(), dist.max()
            return (dist - d_min) / (d_max - d_min + 1e-8)
        elif self.normalization == 'none':
            return dist
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

    def soft_rank(self, dist: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Compute soft ranks using softmax.
        For each row, compute the relative ranking of distances.
        """
        # Negative because we want smaller distances to have higher "rank"
        return F.softmax(-dist / temperature, dim=-1)

    def forward(self, z: Tensor, concepts: Tensor) -> Tensor:
        """
        Compute topological consistency loss.

        Args:
            z: Latent representations [B, D] or expected concepts [B, N]
            concepts: True concept values [B, N]

        Returns:
            Scalar loss value
        """
        # Compute pairwise distances
        z_dist = self.compute_pairwise_distance(z, self.distance_metric)
        c_dist = self.compute_pairwise_distance(concepts.float(), 'l2')  # Always L2 for concepts

        if self.use_soft_rank:
            # Compare rank orderings (more robust to scale differences)
            z_rank = self.soft_rank(z_dist)
            c_rank = self.soft_rank(c_dist)
            topo_loss = F.mse_loss(z_rank, c_rank)
        else:
            # Compare normalized distances
            z_dist_norm = self.normalize_distances(z_dist)
            c_dist_norm = self.normalize_distances(c_dist)
            topo_loss = F.mse_loss(z_dist_norm, c_dist_norm)

        return topo_loss


class MNISTConceptEncoder(nn.Module):
    """Encodes MNIST images to concept probability distributions.

    Supports:
    - HalfMNIST: (B, 1, 28, 56) -> split into 2 images of 28x28
    - XOR/MNIST-EO: (B, 1, 28, 112) -> split into 4 images of 28x28
    """

    def __init__(self, n_images: int, n_values: int, hidden_dim: int = 256, use_shared_head: bool = True):
        super().__init__()
        self.n_images = n_images
        self.n_values = n_values
        self.use_shared_head = use_shared_head

        # Shared CNN encoder for 28x28 grayscale images
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

        # Classification head: shared (better) or separate per image
        if use_shared_head:
            # Single shared head for all concept positions (better data efficiency)
            self.head = nn.Linear(hidden_dim, n_values)
        else:
            # Separate head for each concept position (original approach)
            self.heads = nn.ModuleList([
                nn.Linear(hidden_dim, n_values) for _ in range(n_images)
            ])

    def forward(self, x_BNX: Tensor) -> Tensor:
        """
        Args:
            x_BNX: (B, 1, 28, W) where W = 28*n_images (concatenated images)
        Returns:
            logits: (B, N, D) where N=n_images, D=n_values
        """
        B = x_BNX.shape[0]

        # Handle concatenated format - split into n_images
        if x_BNX.dim() == 4:
            W = x_BNX.shape[-1]
            img_width = W // self.n_images
            # Reshape: (B, 1, 28, n*28) -> (B, n, 1, 28, 28)
            x_BNX = x_BNX.view(B, 1, 28, self.n_images, img_width).permute(0, 3, 1, 2, 4)

        if self.use_shared_head:
            # Batch all images together for efficiency with shared head
            x_flat = x_BNX.reshape(B * self.n_images, 1, 28, 28)  # (B*n_images, 1, 28, 28)
            h_flat = self.encoder(x_flat)  # (B*n_images, hidden)
            logits_flat = self.head(h_flat)  # (B*n_images, D)
            logits = logits_flat.view(B, self.n_images, self.n_values)  # (B, N, D)
        else:
            # Process each image with separate heads
            logits_list = []
            for i in range(self.n_images):
                x_i = x_BNX[:, i]  # (B, 1, 28, 28)
                if x_i.dim() == 3:
                    x_i = x_i.unsqueeze(1)
                h_i = self.encoder(x_i)  # (B, hidden)
                logits_i = self.heads[i](h_i)  # (B, D)
                logits_list.append(logits_i)
            logits = torch.stack(logits_list, dim=1)  # (B, N, D)

        return logits


# Keep old name for backwards compatibility
ConceptEncoder = MNISTConceptEncoder


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) following BEARS implementation.

    Args:
        probs: (N, C) probability predictions for C classes, or (N,) for binary
        targets: (N,) ground truth labels
        n_bins: Number of bins for calibration

    Returns:
        ECE value
    """
    if probs.ndim == 1:
        # Binary case: convert to 2-class format
        probs = np.stack([1 - probs, probs], axis=-1)

    # Get predicted class and confidence (max probability)
    pred_classes = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)
    accuracies = (pred_classes == targets).astype(float)

    # Compute ECE using binning
    n_samples = len(targets)
    if n_samples == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count_in_bin = in_bin.sum()

        if count_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * count_in_bin / n_samples

    return ece

class GenerativeNeSy(nn.Module):
    """
    Generative NeSy with semantic loss and entropy regularization.

    Key ideas:
    1. Use semantic loss for constraint satisfaction
    2. Use entropy regularization to avoid shortcuts (conditional or unconditional)
    3. Use topological consistency to enforce geometric structure (NEW)

    Entropy options:
    - conditional=True:  H(w|x,y) - entropy over valid w's for given y (better OOD)
    - conditional=False: H(w|x)   - entropy over all w's (better ID concept acc)

    Topology options:
    - use_topology=True: Add topological loss to enforce latent geometry matches concept geometry
    - topology_weight: Weight for topology loss (default: 1.0)
    - topology_mode: 'expected' (use expected concepts) or 'logits' (use raw logits)
    """

    def __init__(
        self,
        n_images: int = 2,
        n_values: int = 5,
        constraint_fn: callable = None,
        entropy_weight: float = 1.6,
        conditional_entropy: bool = True,
        encoder: nn.Module = None,
        use_shared_head: bool = True,
        # Topology options
        use_topology: bool = False,
        topology_weight: float = 1.0,
        topology_mode: str = 'expected',  # 'expected', 'logits', or 'probs'
        topology_distance: str = 'l2',
        topology_normalization: str = 'mean',
        topology_soft_rank: bool = False,
    ):
        super().__init__()
        self.n_images = n_images
        self.n_values = n_values
        self.constraint_fn = constraint_fn or (lambda w: w.sum(dim=-1, keepdim=True))
        self.entropy_weight = entropy_weight
        self.conditional_entropy = conditional_entropy

        # Topology settings
        self.use_topology = use_topology
        self.topology_weight = topology_weight
        self.topology_mode = topology_mode

        # Use provided encoder or create default MNIST encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = MNISTConceptEncoder(n_images, n_values, use_shared_head=use_shared_head)

        # Topology loss module
        if use_topology:
            self.topo_loss_fn = TopologicalConsistencyLoss(
                distance_metric=topology_distance,
                normalization=topology_normalization,
                use_soft_rank=topology_soft_rank,
            )

        # Pre-compute all valid w combinations
        self._init_valid_assignments()

    def _init_valid_assignments(self):
        """Pre-compute all possible w assignments and their y values."""
        indices = torch.cartesian_prod(*[torch.arange(self.n_values) for _ in range(self.n_images)])
        y_values = self.constraint_fn(indices)
        self.register_buffer('all_w', indices)  # (D^N, N)
        self.register_buffer('all_y', y_values)  # (D^N, Y)

    def compute_semantic_loss_and_entropy(
        self,
        logits_BND: Tensor,
        y_BY: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute semantic loss and conditional entropy exactly.

        Args:
            logits_BND: Logits for each concept, (B, N, D)
            y_BY: Target labels, (B, Y)

        Returns:
            loss_B: Semantic loss per sample
            entropy_B: Conditional entropy per sample
        """
        B, N, D = logits_BND.shape
        device = logits_BND.device

        # Softmax to get probabilities
        probs_BND = F.softmax(logits_BND, dim=-1)

        # Compute log p(w) for all possible w combinations
        # all_w: (M, N) where M = D^N
        M = self.all_w.shape[0]

        # log p(w) = sum_i log p(w_i)
        log_probs_BM = torch.zeros(B, M, device=device)
        for i in range(N):
            log_probs_BM += torch.log(probs_BND[:, i, self.all_w[:, i]] + 1e-10)

        probs_BM = torch.exp(log_probs_BM)  # (B, M)

        # Mask for w's that satisfy constraint
        # all_y: (M, Y)
        mask_BM = (self.all_y.unsqueeze(0) == y_BY.unsqueeze(1)).all(dim=-1).float()  # (B, M)

        # p(y|x) = sum_w p(w|x) * 1{f(w)=y}
        p_y_B = (probs_BM * mask_BM).sum(dim=-1) + 1e-10

        # Semantic loss = -log p(y|x)
        loss_B = -torch.log(p_y_B)

        # p(w|x,y) = p(w|x) * 1{f(w)=y} / p(y|x)
        p_w_given_y_BM = (probs_BM * mask_BM) / p_y_B.unsqueeze(-1)

        # Conditional entropy H(w|x,y) - entropy over valid w's only
        cond_entropy_B = -torch.sum(
            p_w_given_y_BM * torch.log(p_w_given_y_BM + 1e-10),
            dim=-1
        ) / N  # Normalize by number of concepts

        # Joint unconditional entropy H(w1,w2|x) - entropy over ALL w pairs (BAD - causes 40% acc)
        joint_entropy_B = -torch.sum(
            probs_BM * torch.log(probs_BM + 1e-10),
            dim=-1
        ) / N  # Normalize by number of concepts

        # MARGINAL unconditional entropy = (1/N) * sum_i H(w_i|x) - entropy per concept independently
        # This is what NeSy DM actually uses! Each concept's distribution is encouraged to be uncertain.
        marginal_entropy_B = -torch.sum(
            probs_BND * torch.log(probs_BND + 1e-10),
            dim=-1  # sum over values D
        ).mean(dim=-1)  # mean over concepts N

        # Select which entropy to use based on setting
        entropy_B = cond_entropy_B if self.conditional_entropy else marginal_entropy_B

        return loss_B, entropy_B, probs_BM, cond_entropy_B, marginal_entropy_B

    def predict_y_from_probs(self, probs_BM: Tensor) -> Tensor:
        """
        Predict y using argmax over p(y|x) = sum_w p(w|x) * 1{f(w)=y}

        This is the CORRECT way to compute y accuracy with high-entropy distributions.
        Independent argmax over marginals is WRONG when the distribution has high entropy.
        """
        B = probs_BM.shape[0]
        device = probs_BM.device

        # Get unique y values
        unique_y = torch.unique(self.all_y)  # (K,) or (K, Y)
        if unique_y.dim() == 1:
            unique_y = unique_y.unsqueeze(-1)
        K = unique_y.shape[0]

        # Compute p(y|x) for each unique y
        p_y_BK = torch.zeros(B, K, device=device)
        for k in range(K):
            mask_M = (self.all_y == unique_y[k]).all(dim=-1)  # (M,)
            p_y_BK[:, k] = probs_BM[:, mask_M].sum(dim=-1)

        # Return argmax y
        best_y_idx = p_y_BK.argmax(dim=-1)  # (B,)
        return unique_y[best_y_idx]  # (B, Y)

    def predict_w_from_probs(self, probs_BM: Tensor, y_BY: Tensor = None) -> Tensor:
        """
        Predict w using argmax over p(w|x) or p(w|x,y).

        If y_BY is provided, predict argmax of p(w|x,y).
        Otherwise, predict argmax of p(w|x).
        """
        if y_BY is not None:
            # Mask for w's that satisfy constraint
            mask_BM = (self.all_y.unsqueeze(0) == y_BY.unsqueeze(1)).all(dim=-1).float()
            probs_BM = probs_BM * mask_BM

        best_w_idx = probs_BM.argmax(dim=-1)  # (B,)
        return self.all_w[best_w_idx]  # (B, N)

    def loss(
        self,
        x_BX: Tensor,
        y_BY: Tensor,
        log: IterNeSyLog = None,
        eval_w_BN: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute training loss."""
        self.train()

        # Get initial logits
        logits_BND = self.encoder(x_BX)

        return self._compute_loss(logits_BND, y_BY, log, eval_w_BN)

    def _compute_loss(
        self,
        logits_BND: Tensor,
        y_BY: Tensor,
        log: IterNeSyLog = None,
        eval_w_BN: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute loss for MNIST-like datasets with exact enumeration."""
        B, N, D = logits_BND.shape

        # Compute loss, entropy, and joint probabilities
        loss_B, entropy_B, probs_BM, cond_ent_B, marginal_ent_B = self.compute_semantic_loss_and_entropy(logits_BND, y_BY)

        # Base loss: semantic loss - entropy regularization
        loss = loss_B.mean() - self.entropy_weight * entropy_B.mean()

        # Topological consistency loss (optional)
        topo_loss_value = 0.0
        if self.use_topology and eval_w_BN is not None:
            # Compute latent representation based on topology_mode
            if self.topology_mode == 'expected':
                # Expected concept values: E[c_i] = sum_j p(c_i=j) * j
                probs_BND = F.softmax(logits_BND, dim=-1)  # (B, N, D)
                concept_values = torch.arange(D, device=logits_BND.device, dtype=torch.float32)
                z = (probs_BND * concept_values).sum(dim=-1)  # (B, N)
            elif self.topology_mode == 'logits':
                # Flatten logits as representation
                z = logits_BND.view(B, -1)  # (B, N*D)
            elif self.topology_mode == 'probs':
                # Flatten probabilities as representation
                probs_BND = F.softmax(logits_BND, dim=-1)
                z = probs_BND.view(B, -1)  # (B, N*D)
            else:
                raise ValueError(f"Unknown topology_mode: {self.topology_mode}")

            # Compute topology loss
            topo_loss = self.topo_loss_fn(z, eval_w_BN.float())
            loss = loss + self.topology_weight * topo_loss
            topo_loss_value = topo_loss.item()

        # Logging
        if log is not None:
            log.n_batches += 1
            log.loss += loss.item()
            log.entropy += entropy_B.mean().item()
            log.cond_entropy += cond_ent_B.mean().item()
            log.uncond_entropy += uncond_ent_B.mean().item()
            log.topo_loss += topo_loss_value

            # Predict y first, then predict w conditioned on predicted y
            # This matches how NeSy diffusion logs: sample w -> compute y from w -> measure accuracy
            y_pred = self.predict_y_from_probs(probs_BM)
            w_pred = self.predict_w_from_probs(probs_BM, y_pred)  # Use PREDICTED y, not true y

            log.accuracy_y += (y_pred == y_BY).float().mean().item()
            if eval_w_BN is not None:
                log.accuracy_w += (w_pred == eval_w_BN).float().mean().item()

        return loss

    @torch.no_grad()
    def predict(self, x_BX: Tensor, y_BY: Tensor = None) -> Tuple[Tensor, Tensor]:
        """Predict concepts and labels using proper joint distribution."""
        self.eval()
        logits_BND = self.encoder(x_BX)

        return self._compute_predict(logits_BND, y_BY)

    def _compute_predict(self, logits_BND: Tensor, y_BY: Tensor = None) -> Tuple[Tensor, Tensor]:
        """Predict for MNIST-like datasets."""
        # Compute joint probabilities p(w|x)
        B, N, D = logits_BND.shape
        probs_BND = F.softmax(logits_BND, dim=-1)

        log_probs_BM = torch.zeros(B, self.all_w.shape[0], device=logits_BND.device)
        for i in range(N):
            log_probs_BM += torch.log(probs_BND[:, i, self.all_w[:, i]] + 1e-10)
        probs_BM = torch.exp(log_probs_BM)

        # Predict y using argmax p(y|x)
        y_pred = self.predict_y_from_probs(probs_BM)

        # Predict w using argmax p(w|x,y) given predicted y
        w_pred = self.predict_w_from_probs(probs_BM, y_pred)
        return w_pred, y_pred


def train_and_evaluate():
    """Train and evaluate the model."""
    import argparse
    from torch.utils.data import DataLoader
    import time
    import wandb

    from expressive.experiments.rsbench.datasets import get_dataset
    from expressive.args import RSBenchArguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfmnist')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_weight', type=float, default=None,
                        help='Entropy weight. Default: 1.6 for conditional, 0.01 for marginal.')
    parser.add_argument('--conditional_entropy', action='store_true', default=True,
                        help='Use H(w|x,y) conditional entropy (default). ~70%% concept acc.')
    parser.add_argument('--marginal_entropy', action='store_true', default=False,
                        help='Use (1/N)*sum H(w_i|x) marginal entropy (like NeSy DM). Better concept acc.')
    parser.add_argument('--use_shared_head', action='store_true', default=True,
                        help='Use shared classification head (default). Better convergence.')
    parser.add_argument('--separate_heads', action='store_true', default=False,
                        help='Use separate heads per concept position.')
    # Topology options
    parser.add_argument('--use_topology', action='store_true', default=False,
                        help='Enable topological consistency loss.')
    parser.add_argument('--topology_weight', type=float, default=1.0,
                        help='Weight for topology loss.')
    parser.add_argument('--topology_mode', type=str, default='expected',
                        choices=['expected', 'logits', 'probs'],
                        help='How to compute latent representation for topology: '
                             'expected (E[c]), logits (raw), probs (softmax).')
    parser.add_argument('--topology_distance', type=str, default='l2',
                        choices=['l2', 'l1', 'cosine'],
                        help='Distance metric for topology loss.')
    parser.add_argument('--topology_normalization', type=str, default='mean',
                        choices=['mean', 'minmax', 'none'],
                        help='Normalization for topology distances.')
    parser.add_argument('--topology_soft_rank', action='store_true', default=False,
                        help='Use soft ranking instead of direct distance comparison.')
    parser.add_argument('--no_entropy', action='store_true', default=False,
                        help='Disable entropy term (use topology only).')
    parser.add_argument('--test_every', type=int, default=10)
    parser.add_argument('--n_values', type=int, default=5)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='nesy-diffusion-ablation')
    args = parser.parse_args()

    # Validate dataset
    if args.dataset not in ['halfmnist', 'shortmnist', 'xor']:
        raise ValueError(f"Unknown dataset: {args.dataset}. Supported: halfmnist, shortmnist, xor")

    # Determine entropy type and head type
    use_conditional = not args.marginal_entropy
    entropy_type = 'conditional' if use_conditional else 'marginal'
    use_shared_head = not args.separate_heads
    head_type = 'shared' if use_shared_head else 'separate'

    # Determine effective entropy weight
    effective_entropy_weight = 0.0 if args.no_entropy else args.entropy_weight

    # Build run name with topology info
    run_name_parts = [f"GenNeSy-{args.dataset}"]
    if not args.no_entropy:
        run_name_parts.append(f"ew{args.entropy_weight}-{entropy_type[:4]}")
    if args.use_topology:
        run_name_parts.append(f"topo{args.topology_weight}-{args.topology_mode[:3]}")
    if args.no_entropy and not args.use_topology:
        run_name_parts.append("baseline")
    run_name = "-".join(run_name_parts)

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            'model': 'GenerativeNeSy',
            'dataset': args.dataset,
            'n_epochs': args.n_epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'entropy_weight': effective_entropy_weight,
            'entropy_type': entropy_type if not args.no_entropy else 'disabled',
            'conditional_entropy': use_conditional,
            'use_shared_head': use_shared_head,
            'head_type': head_type,
            'n_values': args.n_values,
            # Topology config
            'use_topology': args.use_topology,
            'topology_weight': args.topology_weight if args.use_topology else 0.0,
            'topology_mode': args.topology_mode,
            'topology_distance': args.topology_distance,
            'topology_normalization': args.topology_normalization,
            'topology_soft_rank': args.topology_soft_rank,
            'no_entropy': args.no_entropy,
        },
        mode="online" if args.use_wandb else "offline",
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    rsbench_args = RSBenchArguments()
    rsbench_args.dataset = args.dataset
    rsbench_args.batch_size = args.batch_size

    dataset = get_dataset(rsbench_args)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    ood_loaders = dataset.get_ood_loaders()

    print(f"Train: {len(train_loader.dataset)}, OOD: {len(ood_loaders[0].dataset) if ood_loaders else 0}")

    # Configure model based on dataset
    if args.dataset == 'halfmnist':
        # HalfMNIST: 2 images, 5 values each (digits 0-4), y = sum
        n_images = 2
        n_values = 5
        constraint_fn = lambda w: w.sum(dim=-1, keepdim=True)
        print(f"Dataset: HalfMNIST - 2 images × 5 concepts, y = c1 + c2")

    elif args.dataset == 'shortmnist':
        # ShortcutMNIST (Even-Odd): 2 images, 10 values each (digits 0-9), y = sum
        # Training has bias: even digits pair with even, odd with odd
        n_images = 2
        n_values = 10
        constraint_fn = lambda w: w.sum(dim=-1, keepdim=True)
        print(f"Dataset: ShortcutMNIST (Even-Odd) - 2 images × 10 concepts, y = c1 + c2")

    elif args.dataset == 'xor':
        # MNIST Even-Odd (XOR): 4 images, 2 values each (even/odd), y = XOR of all
        n_images = 4
        n_values = 2
        # XOR constraint: y = (c1 XOR c2 XOR c3 XOR c4)
        constraint_fn = lambda w: ((w.sum(dim=-1) % 2)).unsqueeze(-1)
        print(f"Dataset: XOR (MNIST Even-Odd) - 4 images × 2 concepts, y = XOR(c1,c2,c3,c4)")

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Supported: halfmnist, shortmnist, xor")

    # Create model
    model = GenerativeNeSy(
        n_images=n_images,
        n_values=n_values,
        constraint_fn=constraint_fn,
        entropy_weight=effective_entropy_weight,
        conditional_entropy=use_conditional,
        use_shared_head=use_shared_head,
        # Topology options
        use_topology=args.use_topology,
        topology_weight=args.topology_weight,
        topology_mode=args.topology_mode,
        topology_distance=args.topology_distance,
        topology_normalization=args.topology_normalization,
        topology_soft_rank=args.topology_soft_rank,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if not args.no_entropy:
        print(f"Entropy: weight={effective_entropy_weight}, type={entropy_type} ({'H(w|x,y)' if use_conditional else 'H(w|x)'})")
    else:
        print(f"Entropy: DISABLED")
    if args.use_topology:
        print(f"Topology: weight={args.topology_weight}, mode={args.topology_mode}, "
              f"distance={args.topology_distance}, norm={args.topology_normalization}, "
              f"soft_rank={args.topology_soft_rank}")
    else:
        print(f"Topology: DISABLED")
    print(f"Head type: {head_type} ({'shared across positions' if use_shared_head else 'separate per position'})")

    # Log model info to wandb (allow_val_change for dataset-specific overrides)
    wandb.config.update({
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'n_images': n_images,
        'n_values': n_values,
    }, allow_val_change=True)

    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    log = IterNeSyLog()

    best_ood_acc = 0.0
    best_val_acc = 0.0

    for epoch in range(args.n_epochs):
        start_time = time.time()

        # Train
        model.train()
        log.reset()

        for batch in train_loader:
            images, labels, concepts = batch
            images = images.to(device)
            labels = labels.to(device)
            concepts = concepts.to(device)

            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)

            optimizer.zero_grad()
            loss = model.loss(images, labels, log, concepts)
            loss.backward()
            optimizer.step()

        n = log.n_batches
        train_stats = {
            'train/loss': log.loss / n,
            'train/entropy': log.entropy / n,
            'train/cond_entropy': log.cond_entropy / n,
            'train/uncond_entropy': log.uncond_entropy / n,
            'train/topo_loss': log.topo_loss / n,
            'train/acc_y': log.accuracy_y / n,
            'train/acc_w': log.accuracy_w / n,
        }

        epoch_time = time.time() - start_time

        # Build print message
        msg = f"Epoch {epoch+1}/{args.n_epochs}: Loss={train_stats['train/loss']:.4f}"
        if not args.no_entropy:
            msg += f", H={train_stats['train/entropy']:.4f}"
        if args.use_topology:
            msg += f", Topo={train_stats['train/topo_loss']:.4f}"
        msg += f", ACC_Y={train_stats['train/acc_y']*100:.1f}%, ACC_W={train_stats['train/acc_w']*100:.1f}%"
        print(msg)

        # Log training stats to wandb
        wandb.log({
            'epoch': epoch + 1,
            'epoch_time': epoch_time,
            **train_stats,
        })

        # Evaluate on validation set
        model.eval()
        val_correct_w = 0
        val_correct_y = 0
        val_total = 0

        for batch in val_loader:
            images, labels, concepts = batch
            images = images.to(device)
            labels = labels.to(device)
            concepts = concepts.to(device)

            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)

            w_pred, y_pred = model.predict(images)
            val_correct_w += (w_pred == concepts).float().mean().item() * images.shape[0]
            val_correct_y += (y_pred == labels).all(dim=-1).float().mean().item() * images.shape[0]
            val_total += images.shape[0]

        val_acc_w = val_correct_w / val_total
        val_acc_y = val_correct_y / val_total

        val_stats = {
            'val/acc_y': val_acc_y,
            'val/acc_w': val_acc_w,
        }

        wandb.log(val_stats)

        if val_acc_w > best_val_acc:
            best_val_acc = val_acc_w

        # Evaluate on OOD set
        if ood_loaders:
            correct_w = 0
            correct_y = 0
            total = 0

            for batch in ood_loaders[0]:
                images, labels, concepts = batch
                images = images.to(device)
                labels = labels.to(device)
                concepts = concepts.to(device)

                if labels.dim() == 1:
                    labels = labels.unsqueeze(-1)

                w_pred, y_pred = model.predict(images)
                correct_w += (w_pred == concepts).float().mean().item() * images.shape[0]
                correct_y += (y_pred == labels).all(dim=-1).float().mean().item() * images.shape[0]
                total += images.shape[0]

            ood_acc_w = correct_w / total
            ood_acc_y = correct_y / total

            wandb.log({
                'ood/acc_y': ood_acc_y,
                'ood/acc_w': ood_acc_w,
            })

            if (epoch + 1) % args.test_every == 0:
                print(f"  Val: ACC_Y={val_acc_y*100:.1f}%, ACC_W={val_acc_w*100:.1f}%")
                print(f"  OOD: ACC_Y={ood_acc_y*100:.1f}%, ACC_W={ood_acc_w*100:.1f}%")

            if ood_acc_w > best_ood_acc:
                best_ood_acc = ood_acc_w
                wandb.run.summary['best_ood_acc_w'] = best_ood_acc
                wandb.run.summary['best_ood_epoch'] = epoch + 1
                if (epoch + 1) % args.test_every == 0:
                    print(f"  *** New best OOD: {best_ood_acc*100:.1f}% ***")
        else:
            # No OOD loaders - print val stats
            if (epoch + 1) % args.test_every == 0:
                print(f"  Val: ACC_Y={val_acc_y*100:.1f}%, ACC_W={val_acc_w*100:.1f}%")

    # Final summary
    if ood_loaders:
        wandb.run.summary['final_ood_acc_w'] = ood_acc_w
    wandb.run.summary['final_val_acc_w'] = val_acc_w
    wandb.run.summary['best_val_acc_w'] = best_val_acc

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    if ood_loaders:
        print(f"Best OOD ACC_W: {best_ood_acc*100:.1f}%")
    print(f"Best Val ACC_W: {best_val_acc*100:.1f}%")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    train_and_evaluate()
