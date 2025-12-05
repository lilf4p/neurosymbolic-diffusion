import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from torch import Tensor


class MNISTEncoder(nn.Module):
    def __init__(self, embedding_size: int, size=16 * 4 * 4):
        super(MNISTEncoder, self).__init__()
        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(size, embedding_size),
        )

    def forward(self, x_B_D_28_28):
        x_BD_1_28_28 = x_B_D_28_28.reshape(-1, 1, 28, 28)
        x_BD_1_W_H = self.encoder(x_BD_1_28_28)
        x_BD_E = x_BD_1_W_H.view(-1, self.size)
        x_BD_E = self.mlp(x_BD_E)
        x_BDE = x_BD_E.view(x_B_D_28_28.shape[0], x_B_D_28_28.shape[1], -1)
        return x_BDE


class SimpleCNNModel(nn.Module):
    """
    A simple CNN-based model for MNIST digit recognition that directly predicts
    digit probabilities without the diffusion process.

    This model:
    1. Encodes each input MNIST digit image using a CNN
    2. Outputs probability distributions over digits (0-9) for each position
    3. Uses cross-entropy loss for training
    4. Can be used with neurosymbolic reasoning by applying symbolic constraints
       to the predicted digit distributions
    """

    def __init__(self, N: int, hidden_size: int = 256):
        """
        Args:
            N: Number of digits per operand (e.g., N=2 means 2-digit numbers)
            hidden_size: Size of the hidden layer in the MLP
        """
        super(SimpleCNNModel, self).__init__()
        self.N = N
        self.n_digits = 2 * N  # Total number of digit positions (2 operands)

        # CNN encoder for each digit (shared weights)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 7x7 -> 3x3
        )

        # Feature size after CNN: 128 * 3 * 3 = 1152
        self.feature_size = 128 * 3 * 3

        # MLP head for digit classification
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 10),  # 10 classes for digits 0-9
        )

    def encode_digits(self, x: Tensor) -> Tensor:
        """
        Encode input MNIST images to digit logits.

        Args:
            x: Input tensor of shape (B, n_digits, 28, 28) or (B, n_digits*28, 28)
               where B is batch size and n_digits is the number of digit positions

        Returns:
            logits: Tensor of shape (B, n_digits, 10) containing logits for each digit
        """
        B = x.shape[0]

        # Reshape input: (B, n_digits, 28, 28) or (B, n_digits*28, 28)
        if len(x.shape) == 3:
            # Input is (B, n_digits*28, 28), reshape to (B, n_digits, 28, 28)
            x = x.view(B, self.n_digits, 28, 28)

        # Process each digit image through the shared CNN
        # Reshape to (B * n_digits, 1, 28, 28) for batch processing
        x_flat = x.view(B * self.n_digits, 1, 28, 28)

        # Encode through CNN
        features = self.encoder(x_flat)
        features = features.view(B * self.n_digits, -1)

        # Classify each digit
        logits = self.classifier(features)

        # Reshape back to (B, n_digits, 10)
        logits = logits.view(B, self.n_digits, 10)

        return logits

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass returning digit probabilities.

        Args:
            x: Input tensor of shape (B, n_digits*28, 28)

        Returns:
            probs: Tensor of shape (B, n_digits, 10) containing probabilities for each digit
        """
        logits = self.encode_digits(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_digit_predictions(self, x: Tensor) -> Tensor:
        """
        Get the most likely digit for each position.

        Args:
            x: Input tensor of shape (B, n_digits*28, 28)

        Returns:
            predictions: Tensor of shape (B, n_digits) with predicted digit values
        """
        probs = self.forward(x)
        predictions = probs.argmax(dim=-1)
        return predictions


class SimpleCNNNeSy(nn.Module):
    """
    Neurosymbolic wrapper for SimpleCNNModel that combines neural predictions
    with symbolic reasoning for the MNIST addition task.

    This model:
    1. Uses a CNN to predict digit distributions
    2. Applies symbolic constraints (addition operation) to compute output predictions
    3. Can use different loss functions: pure neural, constrained, or joint
    """

    def __init__(self, args):
        super(SimpleCNNNeSy, self).__init__()
        self.N = args.N
        self.hidden_size = getattr(args, 'cnn_hidden_size', 256)
        self.cnn = SimpleCNNModel(args.N, self.hidden_size)
        self.args = args

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning digit probabilities."""
        return self.cnn(x)

    def loss(self, x: Tensor, y: Tensor, log, w_labels: Optional[Tensor] = None) -> Tensor:
        """
        Compute the loss for training.

        Args:
            x: Input tensor of shape (B, n_digits*28, 28)
            y: Target sum in digit form (B, N+1) - the result of adding the two N-digit numbers
            log: Logger for tracking metrics
            w_labels: Ground truth digit labels (B, n_digits) - the actual digits in the images

        Returns:
            loss: Scalar loss value
        """
        logits = self.cnn.encode_digits(x)  # (B, n_digits, 10)

        if w_labels is not None:
            # Supervised loss on individual digits (concept supervision)
            # w_labels shape: (B, n_digits)
            B, n_digits, _ = logits.shape
            loss = F.cross_entropy(
                logits.view(B * n_digits, 10),
                w_labels.view(B * n_digits).long()
            )

            # Log accuracy on digit prediction
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                digit_acc = (preds == w_labels).float().mean()
                log.w_acc_train += digit_acc.item()
        else:
            # TODO: Implement loss computation using symbolic constraints when no concept supervision is available
            # If no concept supervision, we would need to use the symbolic constraint
            # For now, just use the digit labels if available
            raise ValueError("SimpleCNNNeSy requires w_labels (digit supervision) for training")

        log.loss += loss.item()
        return loss

    def evaluate(self, x: Tensor, y: Tensor, w_labels: Tensor, log) -> Dict[str, Tensor]:
        """
        Evaluate the model on a batch.

        Args:
            x: Input tensor of shape (B, n_digits*28, 28)
            y: Target sum in digit form (B, N+1)
            w_labels: Ground truth digit labels (B, n_digits)
            log: Logger for tracking metrics

        Returns:
            result_dict: Dictionary containing predictions and ground truth
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)  # (B, n_digits, 10)
            preds = probs.argmax(dim=-1)  # (B, n_digits)

            # Compute digit accuracy (concept accuracy)
            digit_acc = (preds == w_labels).float().mean()
            log.w_acc_avg += digit_acc.item()

            # Compute sum from predicted digits
            pred_sum = self._compute_sum_from_digits(preds)
            target_sum = self._digits_to_number(y)

            # Compute sum accuracy
            sum_acc = (pred_sum == target_sum).float().mean()
            log.y_acc_avg += sum_acc.item()

            # For compatibility with the diffusion model's evaluation
            log.y_acc_top += sum_acc.item()

            # Store predictions for all prediction type keys expected by logger
            result_dict = {
                "LABELS": y,
                "CONCEPTS": w_labels,
                "w_MM": preds,  # Marginal mode (same as preds for deterministic model)
                "w_TM": preds,  # True mode (same as preds for deterministic model)
            }

            # Compute y predictions
            pred_y = self._compute_sum_digits_from_digits(preds)
            result_dict["y_MMf"] = pred_y
            result_dict["y_TMf"] = pred_y
            result_dict["y_fMM"] = pred_y
            result_dict["y_fTM"] = pred_y

            # For compatibility, add sample dimensions
            result_dict["W_SAMPLES"] = preds.unsqueeze(0)
            result_dict["Y_SAMPLES"] = pred_y.unsqueeze(0)

            # Update pred_types dict in log
            for key in ["w_MM", "w_TM"]:
                if hasattr(log, 'pred_types') and key in log.pred_types:
                    log.pred_types[key] += (preds == w_labels).float().mean().item()
            for key in ["y_MMf", "y_TMf", "y_fMM", "y_fTM"]:
                if hasattr(log, 'pred_types') and key in log.pred_types:
                    log.pred_types[key] += sum_acc.item()

        return result_dict

    def _compute_sum_from_digits(self, digits: Tensor) -> Tensor:
        """
        Compute the sum from predicted digits.

        Args:
            digits: Tensor of shape (B, n_digits) with digit values

        Returns:
            sums: Tensor of shape (B,) with computed sums
        """
        B = digits.shape[0]
        N = self.N

        # Split into first and second number
        first_digits = digits[:, :N]  # (B, N)
        second_digits = digits[:, N:]  # (B, N)

        # Convert to numbers
        powers = torch.tensor([10 ** (N - 1 - i) for i in range(N)],
                            device=digits.device, dtype=digits.dtype)
        first_number = (first_digits * powers).sum(dim=-1)
        second_number = (second_digits * powers).sum(dim=-1)

        return first_number + second_number

    def _digits_to_number(self, digits: Tensor) -> Tensor:
        """
        Convert digit tensor to number.

        Args:
            digits: Tensor of shape (B, num_digits)

        Returns:
            numbers: Tensor of shape (B,)
        """
        num_digits = digits.shape[-1]
        powers = torch.tensor([10 ** (num_digits - 1 - i) for i in range(num_digits)],
                            device=digits.device, dtype=digits.dtype)
        return (digits * powers).sum(dim=-1)

    def _compute_sum_digits_from_digits(self, digits: Tensor) -> Tensor:
        """
        Compute the sum and return as digits.

        Args:
            digits: Tensor of shape (B, n_digits) with digit values

        Returns:
            sum_digits: Tensor of shape (B, N+1) with sum as digits
        """
        sums = self._compute_sum_from_digits(digits)
        N = self.N
        num_result_digits = N + 1

        # Convert sum to digits
        result = torch.zeros(sums.shape[0], num_result_digits,
                           device=digits.device, dtype=digits.dtype)
        for i in range(num_result_digits):
            result[:, i] = (sums // (10 ** (num_result_digits - 1 - i))) % 10

        return result

    def parameters(self):
        """Return model parameters for optimizer."""
        return self.cnn.parameters()

    def state_dict(self):
        """Return model state dict for saving."""
        return self.cnn.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict."""
        return self.cnn.load_state_dict(state_dict)
