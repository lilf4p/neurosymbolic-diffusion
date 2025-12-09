"""
Suggested encoder architecture for semantic loss on MNIST.

This follows the recommendations:
- Conv(5×5, 32), ReLU, MaxPool(2)
- Conv(5×5, 64), ReLU, MaxPool(2)
- Flatten → FC 128, ReLU
- Head per concept: FC 64 → ReLU → C-way softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTSuggestedEncoder(nn.Module):
    """
    Encoder architecture following the suggested design for semantic loss.

    Architecture:
    - Conv(5×5, 32), ReLU, MaxPool(2)
    - Conv(5×5, 64), ReLU, MaxPool(2)
    - Flatten → FC 128, ReLU
    - Head: FC 64 → ReLU → FC c_dim (softmax applied externally)
    """

    def __init__(
        self, img_channels=1, c_dim=10, dropout=0.5, n_images=2
    ):
        super(MNISTSuggestedEncoder, self).__init__()

        self.img_channels = img_channels
        self.c_dim = c_dim
        self.n_images = n_images

        # Conv layers as suggested
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,  # Same padding to preserve spatial dims before pooling
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)

        # After two MaxPool(2) on 28x28: 28 -> 14 -> 7
        # So flattened size = 64 * 7 * 7 = 3136
        self.flatten_size = 64 * 7 * 7

        # FC 128 as suggested
        self.fc1 = nn.Linear(self.flatten_size, 128)

        # Concept head: FC 64 → ReLU → FC c_dim
        self.fc_head1 = nn.Linear(128, 64)
        self.fc_head2 = nn.Linear(64, c_dim)

        # For compatibility with existing code that expects mu/logvar
        self.dense_mu = nn.Linear(128, 16 * c_dim)
        self.dense_logvar = nn.Linear(128, 16 * c_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input image [B, 1, 28, 28] (single digit crop)

        Returns:
            tuple: (concept_logits, mu, logvar)
                - concept_logits: [B, 1, c_dim] (to match original encoder format)
                - mu: [B, c_dim, 16] (for compatibility)
                - logvar: [B, c_dim, 16] (for compatibility)
        """
        # Conv block 1: Conv(5×5, 32) → ReLU → MaxPool(2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Conv block 2: Conv(5×5, 64) → ReLU → MaxPool(2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten → FC 128 → ReLU
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(x))

        # Concept head: FC 64 → ReLU → FC c_dim
        head = F.relu(self.fc_head1(features))
        concept_logits = self.fc_head2(head)

        # For compatibility with existing code - reshape to [B, 1, c_dim]
        concept_logits = concept_logits.unsqueeze(1)

        # For compatibility with existing code
        mu = self.dense_mu(features).view(-1, self.c_dim, 16)
        logvar = self.dense_logvar(features).view(-1, self.c_dim, 16)

        return concept_logits, mu, logvar


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the encoder
    encoder = MNISTSuggestedEncoder(c_dim=5)
    print(f"Suggested Encoder parameters: {count_parameters(encoder):,}")

    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    logits, mu, logvar = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Compare with original
    from addmnist_single import MNISTSingleEncoder
    original = MNISTSingleEncoder(c_dim=5)
    print(f"\nOriginal Encoder parameters: {count_parameters(original):,}")
