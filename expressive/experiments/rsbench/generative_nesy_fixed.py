"""

NeSy CNN model with Semantic Loss and Entropy Regularization.

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
    marginal_entropy: float = 0.0
    loss: float = 0.0
    n_batches: int = 0

    def reset(self):
        self.accuracy_w = 0.0
        self.accuracy_y = 0.0
        self.entropy = 0.0
        self.cond_entropy = 0.0
        self.marginal_entropy = 0.0
        self.loss = 0.0
        self.n_batches = 0


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
    2. Use entropy regularization to avoid shortcuts

    Entropy options:
    - conditional=True:  H(w|x,y) - entropy over valid w's for given y (~70% concept acc)
                         Good for OOD generalization but limits concept accuracy for ambiguous sums.
    - conditional=False: (1/N) * sum_i H(w_i|x) - MARGINAL entropy per concept (like NeSy DM!)
                         This is what NeSy DM actually uses. Encourages each digit prediction to be
                         uncertain independently, without encouraging joint correlation.
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
    ):
        super().__init__()
        self.n_images = n_images
        self.n_values = n_values
        self.constraint_fn = constraint_fn or (lambda w: w.sum(dim=-1, keepdim=True))
        self.entropy_weight = entropy_weight
        self.conditional_entropy = conditional_entropy

        # Use provided encoder or create default MNIST encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = MNISTConceptEncoder(n_images, n_values, use_shared_head=use_shared_head)

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
        # Compute loss, entropy, and joint probabilities
        loss_B, entropy_B, probs_BM, cond_ent_B, marginal_ent_B = self.compute_semantic_loss_and_entropy(logits_BND, y_BY)

        # Total loss (entropy_B is already selected based on self.conditional_entropy)
        loss = loss_B.mean() - self.entropy_weight * entropy_B.mean()

        # Logging
        if log is not None:
            log.n_batches += 1
            log.loss += loss.item()
            log.entropy += entropy_B.mean().item()
            log.cond_entropy += cond_ent_B.mean().item()
            log.marginal_entropy += marginal_ent_B.mean().item()

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

    # Set default entropy weight based on entropy type
    # Conditional entropy H(w|x,y) has smaller range → use larger weight (1.6)
    # Marginal entropy H(w_i|x) has larger range → use smaller weight (0.01)
    if args.entropy_weight is None:
        args.entropy_weight = 1.6 if use_conditional else 0.01
        print(f"Using default entropy_weight={args.entropy_weight} for {entropy_type} entropy")

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        name=f"GenNeSy-{args.dataset}-ew{args.entropy_weight}-{entropy_type[:4]}-{head_type[:3]}",
        config={
            'model': 'GenerativeNeSy',
            'dataset': args.dataset,
            'n_epochs': args.n_epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'entropy_weight': args.entropy_weight,
            'entropy_type': entropy_type,
            'conditional_entropy': use_conditional,
            'use_shared_head': use_shared_head,
            'head_type': head_type,
            'n_values': args.n_values,
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
        entropy_weight=args.entropy_weight,
        conditional_entropy=use_conditional,
        use_shared_head=use_shared_head,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Entropy type: {entropy_type} ({'H(w|x,y)' if use_conditional else 'H(w|x)'})")
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
            'train/marginal_entropy': log.marginal_entropy / n,
            'train/acc_y': log.accuracy_y / n,
            'train/acc_w': log.accuracy_w / n,
        }

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{args.n_epochs}: Loss={train_stats['train/loss']:.4f}, "
              f"H={train_stats['train/entropy']:.4f} (cond={train_stats['train/cond_entropy']:.4f}, marg={train_stats['train/marginal_entropy']:.4f}), "
              f"ACC_Y={train_stats['train/acc_y']*100:.1f}%, ACC_W={train_stats['train/acc_w']*100:.1f}%")

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
