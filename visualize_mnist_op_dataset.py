"""
Visualization of the MNIST Operation Dataset used in mnistop experiments.

This script visualizes the dataset used for training the neurosymbolic diffusion model.
The dataset consists of MNIST digit pairs with their sum as the label.
"""

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import math

from expressive.experiments.mnist_op.data import (
    create_nary_multidigit_operation,
    get_mnist_op_dataloaders,
    MNISTOperationDataset,
)


def visualize_mnist_op_samples(n_samples: int = 10, N: int = 2, op_name: str = "sum"):
    """
    Visualize samples from the MNIST operation dataset.

    Args:
        n_samples: Number of samples to visualize
        N: Number of digits per operand
        op_name: Operation type ("sum" or "product")
    """
    # Setup operation
    arity = 2
    digits_per_number = N
    n_operands = arity * digits_per_number

    bin_op = sum if op_name == "sum" else math.prod if op_name == "product" else None
    op = create_nary_multidigit_operation(arity, bin_op)

    # Get dataloaders
    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=1000,
        count_val=200,
        count_test=500,
        batch_size=n_samples,
        n_operands=n_operands,
        op=op,
        shuffle=False,
    )

    # Get a batch of samples
    batch = next(iter(train_loader))

    # Parse the batch structure:
    # batch[0:2*N] = images (mn_digits)
    # batch[2*N:-1] = individual digit labels (label_digits)
    # batch[-1] = operation result label

    mn_digits = batch[:2*N]  # List of N*2 image tensors
    label_digits = batch[2*N:-1]  # List of N+1 label tensors (for the result)
    label = batch[-1]  # Operation result

    print(f"Dataset Configuration:")
    print(f"  - Number of digits per operand (N): {N}")
    print(f"  - Operation: {op_name}")
    print(f"  - Number of operands: {arity}")
    print(f"  - Total digits per sample: {n_operands}")
    print(f"  - Batch structure:")
    print(f"    * Images per sample: {len(mn_digits)} (each {mn_digits[0].shape})")
    print(f"    * Individual digit labels: {len(label_digits)}")
    print(f"    * Result label shape: {label.shape}")
    print()

    # Create visualization
    fig, axes = plt.subplots(n_samples, n_operands + 1, figsize=(3 * (n_operands + 2), 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for sample_idx in range(n_samples):
        # Extract individual digit labels for this sample
        first_number_digits = [int(mn_digits[i][sample_idx].argmax() if isinstance(mn_digits[i][sample_idx], torch.Tensor) and mn_digits[i][sample_idx].dim() > 0 else 0) for i in range(N)]
        second_number_digits = [int(mn_digits[i][sample_idx].argmax() if isinstance(mn_digits[i][sample_idx], torch.Tensor) and mn_digits[i][sample_idx].dim() > 0 else 0) for i in range(N, 2*N)]

        # Get actual digit labels from the dataset
        digit_labels_first = []
        digit_labels_second = []

        # The label_digits in the batch are the individual MNIST labels
        for i in range(N):
            # Find the label by looking at which index in label_digits corresponds to this digit
            digit_labels_first.append(int(label_digits[i][sample_idx]) if i < len(label_digits) else "?")
        for i in range(N):
            digit_labels_second.append(int(label_digits[N + i][sample_idx]) if N + i < len(label_digits) else "?")

        # Plot each digit image
        for digit_idx in range(n_operands):
            ax = axes[sample_idx, digit_idx]
            img = mn_digits[digit_idx][sample_idx].squeeze()

            # Denormalize for visualization
            img = img * 0.3081 + 0.1307

            ax.imshow(img, cmap='gray')
            ax.axis('off')

            if digit_idx < N:
                # First number
                if digit_idx == 0:
                    ax.set_title(f"1st Number\nDigit {digit_idx+1}", fontsize=10)
                else:
                    ax.set_title(f"Digit {digit_idx+1}", fontsize=10)
            else:
                # Second number
                if digit_idx == N:
                    ax.set_title(f"2nd Number\nDigit {digit_idx-N+1}", fontsize=10)
                else:
                    ax.set_title(f"Digit {digit_idx-N+1}", fontsize=10)

        # Show the result in the last column
        ax = axes[sample_idx, n_operands]
        ax.text(0.5, 0.5, f"= {int(label[sample_idx])}", fontsize=20, ha='center', va='center',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.axis('off')
        if sample_idx == 0:
            ax.set_title("Result", fontsize=10)

    plt.suptitle(f"MNIST Operation Dataset: {N}-digit Addition\n"
                 f"Each row shows {n_operands} digit images forming two {N}-digit numbers and their sum",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("mnist_op_dataset_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nVisualization saved to: mnist_op_dataset_visualization.png")


def visualize_detailed_samples(n_samples: int = 5, N: int = 2):
    """
    Show detailed breakdown of samples with digit-level information.
    """
    arity = 2
    n_operands = arity * N
    op = create_nary_multidigit_operation(arity, sum)

    train_loader, _, _ = get_mnist_op_dataloaders(
        count_train=1000,
        count_val=200,
        count_test=500,
        batch_size=n_samples,
        n_operands=n_operands,
        op=op,
        shuffle=False,
    )

    batch = next(iter(train_loader))
    mn_digits = batch[:2*N]
    label_digits = batch[2*N:-1]
    result_label = batch[-1]

    print("\n" + "="*60)
    print("DETAILED SAMPLE BREAKDOWN")
    print("="*60)

    for sample_idx in range(n_samples):
        print(f"\n--- Sample {sample_idx + 1} ---")

        # Build first number
        first_digits = [int(label_digits[i][sample_idx]) for i in range(N)]
        first_number = sum(d * (10 ** (N - 1 - i)) for i, d in enumerate(first_digits))

        # Build second number
        second_digits = [int(label_digits[N + i][sample_idx]) for i in range(N)]
        second_number = sum(d * (10 ** (N - 1 - i)) for i, d in enumerate(second_digits))

        result = int(result_label[sample_idx])

        print(f"  First number:  {''.join(map(str, first_digits))} = {first_number}")
        print(f"  Second number: {''.join(map(str, second_digits))} = {second_number}")
        print(f"  Sum: {first_number} + {second_number} = {result}")

        # Verify
        expected = first_number + second_number
        if expected == result:
            print(f"  ✓ Verified correct!")
        else:
            print(f"  ✗ Mismatch! Expected {expected}")


def visualize_dataset_statistics(N: int = 2):
    """
    Show dataset statistics and distribution.
    """
    arity = 2
    n_operands = arity * N
    op = create_nary_multidigit_operation(arity, sum)

    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=5000,
        count_val=1000,
        count_test=1000,
        batch_size=256,
        n_operands=n_operands,
        op=op,
        shuffle=False,
    )

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    # Collect all results
    all_results = []
    all_first_numbers = []
    all_second_numbers = []

    for batch in train_loader:
        label_digits = batch[2*N:-1]
        result_label = batch[-1]

        for sample_idx in range(len(result_label)):
            first_digits = [int(label_digits[i][sample_idx]) for i in range(N)]
            first_number = sum(d * (10 ** (N - 1 - i)) for i, d in enumerate(first_digits))

            second_digits = [int(label_digits[N + i][sample_idx]) for i in range(N)]
            second_number = sum(d * (10 ** (N - 1 - i)) for i, d in enumerate(second_digits))

            all_first_numbers.append(first_number)
            all_second_numbers.append(second_number)
            all_results.append(int(result_label[sample_idx]))

    print(f"\nTraining set size: {len(all_results)} samples")
    print(f"Validation set size: {len(val_loader.dataset)} samples")
    print(f"Test set size: {len(test_loader.dataset)} samples")

    print(f"\nFirst number range: [{min(all_first_numbers)}, {max(all_first_numbers)}]")
    print(f"Second number range: [{min(all_second_numbers)}, {max(all_second_numbers)}]")
    print(f"Result range: [{min(all_results)}, {max(all_results)}]")

    print(f"\nMean first number: {sum(all_first_numbers)/len(all_first_numbers):.2f}")
    print(f"Mean second number: {sum(all_second_numbers)/len(all_second_numbers):.2f}")
    print(f"Mean result: {sum(all_results)/len(all_results):.2f}")

    # Plot distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(all_first_numbers, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('First Number Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of First Numbers')

    axes[1].hist(all_second_numbers, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Second Number Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Second Numbers')

    axes[2].hist(all_results, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[2].set_xlabel('Sum Result')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Sum Results')

    plt.suptitle(f'MNIST Operation Dataset Statistics (N={N})', fontsize=14)
    plt.tight_layout()
    plt.savefig("mnist_op_statistics.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nStatistics plot saved to: mnist_op_statistics.png")


if __name__ == "__main__":
    # Default configuration matching the experiment
    N = 2  # Number of digits per operand (you can change this to 1, 2, 3, etc.)

    print("="*60)
    print("MNIST Operation Dataset Visualization")
    print("="*60)
    print(f"\nThis dataset is used to train a neurosymbolic diffusion model")
    print(f"to learn addition of {N}-digit numbers from MNIST images.\n")

    # Visualize samples
    visualize_mnist_op_samples(n_samples=8, N=N, op_name="sum")

    # Show detailed breakdown
    visualize_detailed_samples(n_samples=5, N=N)

    # Show statistics
    visualize_dataset_statistics(N=N)
