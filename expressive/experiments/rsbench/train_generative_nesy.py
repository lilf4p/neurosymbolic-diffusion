"""
Training script for Generative NeSy model on HalfMNIST.

This tests the hypothesis that NeSyDM's key components are:
1. Generative sampling process
2. RLOO gradient estimator
3. Entropy regularization

WITHOUT needing:
- Diffusion process
- Denoising objective
"""

import argparse
import os
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from expressive.experiments.rsbench.datasets import get_dataset
from expressive.experiments.rsbench.generative_nesy import (
    GenerativeNeSy,
    GenNeSyLog,
    create_addition_constraint,
)
from expressive.args import RSBenchArguments
from expressive.util import get_device


class GenerativeNeSyForRSBench(GenerativeNeSy):
    """
    GenerativeNeSy adapted for RSBench datasets (HalfMNIST, PermutedHalfMNIST).

    Handles the specific input format and dataset structure.
    """

    def __init__(
        self,
        args: RSBenchArguments,
        n_images: int = 2,
        c_split: list = None,
        n_values: int = 5,
    ):
        if c_split is None:
            c_split = [1, 1]

        # Create constraint function for addition
        constraint_fn = create_addition_constraint(n_values)

        super().__init__(
            n_images=n_images,
            c_split=c_split,
            n_values=n_values,
            constraint_fn=constraint_fn,
            n_samples=getattr(args, 'n_samples', 16),
            entropy_weight=getattr(args, 'entropy_weight', 1.6),
            hidden_dim=256,
            temperature=getattr(args, 'temperature', 1.0),
        )

        self.args = args

    def forward_batch(self, batch, device, log=None):
        """Process a batch from the dataloader."""
        images, labels, concepts = batch
        images = images.to(device)
        labels = labels.to(device)
        concepts = concepts.to(device)

        # Reshape images if needed
        # Expected format: (B, N, C, H, W) where N=2 images
        if images.dim() == 4 and images.shape[1] == 1:
            # (B, 1, 28, 56) -> need to split into two 28x28 images
            B, C, H, W = images.shape
            if W == 56:
                images = images.view(B, C, H, 2, 28).permute(0, 3, 1, 2, 4)  # (B, 2, C, H, 28)
            elif H == 56:
                images = images.view(B, C, 2, 28, W).permute(0, 2, 1, 3, 4)  # (B, 2, C, 28, W)

        # Labels format: (B,) -> (B, 1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        return self.loss(images, labels, log, concepts)


def train_epoch(model, train_loader, optimizer, device, log):
    """Train for one epoch."""
    model.train()
    log.reset()

    for batch in train_loader:
        optimizer.zero_grad()
        loss = model.forward_batch(batch, device, log)
        loss.backward()
        optimizer.step()

    # Average stats
    n = log.n_batches
    return {
        'loss': log.total_loss / n,
        'rloo_loss': log.rloo_loss / n,
        'entropy': log.entropy / n,
        'avg_reward': log.avg_reward / n,
        'accuracy_y': log.accuracy_y / n,
        'accuracy_w': log.accuracy_w / n,
    }


@torch.no_grad()
def evaluate(model, data_loader, device, prefix=""):
    """Evaluate model on a dataset."""
    model.eval()

    all_w_pred = []
    all_w_true = []
    all_y_pred = []
    all_y_true = []

    for batch in data_loader:
        images, labels, concepts = batch
        images = images.to(device)
        labels = labels.to(device)
        concepts = concepts.to(device)

        # Reshape images if needed
        if images.dim() == 4 and images.shape[1] == 1:
            B, C, H, W = images.shape
            if W == 56:
                images = images.view(B, C, H, 2, 28).permute(0, 3, 1, 2, 4)
            elif H == 56:
                images = images.view(B, C, 2, 28, W).permute(0, 2, 1, 3, 4)

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        w_pred, y_pred = model.predict(images)

        all_w_pred.append(w_pred.cpu())
        all_w_true.append(concepts.cpu())
        all_y_pred.append(y_pred.cpu())
        all_y_true.append(labels.cpu())

    all_w_pred = torch.cat(all_w_pred, dim=0)
    all_w_true = torch.cat(all_w_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)

    # Compute metrics
    accuracy_y = (all_y_pred == all_y_true).float().mean().item()
    accuracy_w = (all_w_pred == all_w_true).float().mean().item()

    # Per-digit accuracy
    accuracy_w_per_digit = [
        (all_w_pred[:, i] == all_w_true[:, i]).float().mean().item()
        for i in range(all_w_pred.shape[1])
    ]

    stats = {
        f'{prefix}accuracy_y': accuracy_y,
        f'{prefix}accuracy_w': accuracy_w,
        f'{prefix}accuracy_w_digit0': accuracy_w_per_digit[0],
        f'{prefix}accuracy_w_digit1': accuracy_w_per_digit[1],
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Train Generative NeSy on HalfMNIST")
    parser.add_argument('--dataset', type=str, default='halfmnist',
                        choices=['halfmnist', 'permutedhalfmnist'])
    parser.add_argument('--digit_permutation', type=str, default='identity')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Number of samples for RLOO')
    parser.add_argument('--entropy_weight', type=float, default=1.6,
                        help='Weight for entropy regularization')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling')
    parser.add_argument('--test_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_values', type=int, default=5,
                        help='Number of possible digit values (5 for HalfMNIST)')
    parser.add_argument('--DEBUG', action='store_true')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create RSBench args for dataset loading
    rsbench_args = RSBenchArguments()
    rsbench_args.dataset = args.dataset
    rsbench_args.digit_permutation = args.digit_permutation
    rsbench_args.batch_size = args.batch_size
    rsbench_args.DEBUG = args.DEBUG

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(rsbench_args)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    ood_loaders = dataset.get_ood_loaders()

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"OOD samples: {len(ood_loaders[0].dataset) if ood_loaders else 0}")

    # Create model
    model = GenerativeNeSyForRSBench(
        args=rsbench_args,
        n_images=2,
        c_split=[1, 1],
        n_values=args.n_values,
    ).to(device)

    # Override entropy weight and n_samples from command line
    model.entropy_weight = args.entropy_weight
    model.n_samples = args.n_samples
    model.temperature = args.temperature

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Entropy weight: {args.entropy_weight}")
    print(f"N samples (RLOO): {args.n_samples}")
    print(f"Temperature: {args.temperature}")

    # Optimizer
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)

    # Training log
    log = GenNeSyLog()

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_ood_acc_w = 0.0

    for epoch in range(args.n_epochs):
        start_time = time.time()

        # Train
        train_stats = train_epoch(model, train_loader, optimizer, device, log)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{args.n_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_stats['loss']:.4f} "
              f"(RLOO: {train_stats['rloo_loss']:.4f}, "
              f"Entropy: {train_stats['entropy']:.4f})")
        print(f"  Train ACC Y: {train_stats['accuracy_y']*100:.1f}% "
              f"ACC W: {train_stats['accuracy_w']*100:.1f}%")

        # Evaluate
        if (epoch + 1) % args.test_every == 0 or epoch == 0:
            val_stats = evaluate(model, val_loader, device, "val_")
            print(f"  Val   ACC Y: {val_stats['val_accuracy_y']*100:.1f}% "
                  f"ACC W: {val_stats['val_accuracy_w']*100:.1f}%")

            if ood_loaders:
                ood_stats = evaluate(model, ood_loaders[0], device, "ood_")
                print(f"  OOD   ACC Y: {ood_stats['ood_accuracy_y']*100:.1f}% "
                      f"ACC W: {ood_stats['ood_accuracy_w']*100:.1f}% "
                      f"<-- KEY METRIC")

                if ood_stats['ood_accuracy_w'] > best_ood_acc_w:
                    best_ood_acc_w = ood_stats['ood_accuracy_w']
                    print(f"  *** New best OOD ACC W: {best_ood_acc_w*100:.1f}% ***")

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    val_stats = evaluate(model, val_loader, device, "val_")
    print(f"Val   ACC Y: {val_stats['val_accuracy_y']*100:.1f}% "
          f"ACC W: {val_stats['val_accuracy_w']*100:.1f}%")
    print(f"  Digit 0: {val_stats['val_accuracy_w_digit0']*100:.1f}% "
          f"Digit 1: {val_stats['val_accuracy_w_digit1']*100:.1f}%")

    if ood_loaders:
        ood_stats = evaluate(model, ood_loaders[0], device, "ood_")
        print(f"OOD   ACC Y: {ood_stats['ood_accuracy_y']*100:.1f}% "
              f"ACC W: {ood_stats['ood_accuracy_w']*100:.1f}%")
        print(f"  Digit 0: {ood_stats['ood_accuracy_w_digit0']*100:.1f}% "
              f"Digit 1: {ood_stats['ood_accuracy_w_digit1']*100:.1f}%")

    print("\n" + "="*60)
    print(f"Best OOD ACC W: {best_ood_acc_w*100:.1f}%")
    print("="*60)

    # Interpretation
    print("\n--- INTERPRETATION ---")
    if best_ood_acc_w >= 0.8:
        print("✓ HIGH OOD accuracy! The model learned the correct concepts.")
        print("  → Generative approach + RLOO + Entropy works WITHOUT diffusion!")
    elif best_ood_acc_w >= 0.5:
        print("~ MEDIUM OOD accuracy. Partial shortcut learning.")
        print("  → Generative approach helps but may need tuning.")
    else:
        print("✗ LOW OOD accuracy. Shortcut learning detected.")
        print("  → The model is using shortcuts like 2→4, 3→1.")
        print("  → May need stronger entropy or more samples.")


if __name__ == "__main__":
    main()
