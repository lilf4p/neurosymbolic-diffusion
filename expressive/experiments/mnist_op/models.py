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

                # Compute and log Y accuracy
                pred_sum = self._compute_sum_digits_from_digits(preds)
                y_acc = (pred_sum == y).all(dim=-1).float().mean()
                if hasattr(log, 'var_accuracy_y'):
                    log.var_accuracy_y += y_acc.item()
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


class GenerativeNeSyMNISTAdd(nn.Module):
    """
    Generative NeSy model for MNIST Addition using Semantic Loss + Entropy Regularization.

    This implementation uses EXACT enumeration over all possible w combinations,
    matching the generative_nesy_fixed.py implementation used for HalfMNIST.

    For MNIST Addition: y = sum of two N-digit numbers represented by 2N images
    Each digit position can take values 0-9 (10 values).

    NOTE: Exact enumeration is only feasible for small N (e.g., N=1 or N=2).
    For N=1: 10^2 = 100 combinations
    For N=2: 10^4 = 10,000 combinations
    For N=3+: Use sampling-based approach instead.
    """

    def __init__(self, args):
        super(GenerativeNeSyMNISTAdd, self).__init__()
        self.N = args.N
        self.n_digits = 2 * args.N  # Total digit positions (2 operands)
        self.n_values = 10  # Each digit can be 0-9
        self.hidden_size = getattr(args, 'cnn_hidden_size', 256)
        self.args = args

        # Hyperparameters
        self.entropy_weight = getattr(args, 'sl_entropy_weight', 0.1)
        # Use conditional entropy by default (like generative_nesy_fixed.py)
        self.conditional_entropy = getattr(args, 'sl_conditional_entropy', True)
        # Use shared head for better convergence (default True)
        self.use_shared_head = getattr(args, 'use_shared_head', True)

        # CNN encoder (shared across digits)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.hidden_size),
            nn.ReLU(),
        )

        # Classification head: shared (better) or separate per digit position
        if self.use_shared_head:
            # Single shared head for all digit positions (better data efficiency)
            self.head = nn.Linear(self.hidden_size, self.n_values)
        else:
            # Separate head for each digit position (original approach)
            self.heads = nn.ModuleList([
                nn.Linear(self.hidden_size, self.n_values) for _ in range(self.n_digits)
            ])

        # Pre-compute all valid w combinations and their y values
        self._init_valid_assignments()

    def _init_valid_assignments(self):
        """Pre-compute all possible w assignments and their y values."""
        # Generate all combinations of digits: (n_values^n_digits, n_digits)
        indices = torch.cartesian_prod(*[torch.arange(self.n_values) for _ in range(self.n_digits)])
        # Compute corresponding y values
        y_values = self._constraint_fn(indices)
        self.register_buffer('all_w', indices)  # (M, n_digits) where M = 10^(2N)
        self.register_buffer('all_y', y_values)  # (M, N+1)

        print(f"GenerativeNeSyMNISTAdd: Pre-computed {len(indices)} w combinations for N={self.N}")

    def _constraint_fn(self, w: Tensor) -> Tensor:
        """
        Compute y = first_number + second_number from digit assignments.

        Args:
            w: (M, n_digits) digit assignments

        Returns:
            y: (M, N+1) sum as digits
        """
        N = self.N
        device = w.device

        # Powers of 10 for converting digits to numbers
        powers = torch.tensor([10 ** (N - 1 - i) for i in range(N)],
                             device=device, dtype=torch.int64)

        # First operand: digits 0 to N-1
        first_number = (w[..., :N].to(torch.int64) * powers).sum(dim=-1)

        # Second operand: digits N to 2N-1
        second_number = (w[..., N:].to(torch.int64) * powers).sum(dim=-1)

        # Sum
        total = first_number + second_number

        # Convert sum to digits (N+1 digits to handle carry)
        num_result_digits = N + 1
        result = torch.zeros(*total.shape, num_result_digits, device=device, dtype=torch.int64)
        for i in range(num_result_digits):
            result[..., i] = (total // (10 ** (num_result_digits - 1 - i))) % 10

        return result

    def encode_digits(self, x: Tensor) -> Tensor:
        """
        Encode MNIST images to digit logits.

        Args:
            x: Input tensor of shape (B, n_digits*28, 28) or (B, n_digits, 28, 28)

        Returns:
            logits: (B, n_digits, 10)
        """
        B = x.shape[0]

        # Reshape input to (B, n_digits, 28, 28)
        if len(x.shape) == 3:
            x = x.view(B, self.n_digits, 28, 28)

        # Process each digit through shared CNN encoder
        if self.use_shared_head:
            # Batch all digits together for efficiency with shared head
            x_flat = x.view(B * self.n_digits, 1, 28, 28)  # (B*n_digits, 1, 28, 28)
            h_flat = self.encoder(x_flat)  # (B*n_digits, hidden_size)
            logits_flat = self.head(h_flat)  # (B*n_digits, 10)
            logits = logits_flat.view(B, self.n_digits, 10)  # (B, n_digits, 10)
        else:
            # Process each digit with separate heads
            logits_list = []
            for i in range(self.n_digits):
                x_i = x[:, i:i+1, :, :]  # (B, 1, 28, 28)
                h_i = self.encoder(x_i)  # (B, hidden_size)
                logits_i = self.heads[i](h_i)  # (B, 10)
                logits_list.append(logits_i)
            logits = torch.stack(logits_list, dim=1)  # (B, n_digits, 10)

        return logits

    def compute_semantic_loss_and_entropy(
        self,
        logits_BND: Tensor,
        y_BY: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute semantic loss and entropy exactly (matching generative_nesy_fixed.py).

        Args:
            logits_BND: Logits for each concept, (B, N, D) where D=10
            y_BY: Target labels (sum as digits), (B, Y) where Y=N+1

        Returns:
            loss_B: Semantic loss per sample
            entropy_B: Selected entropy (conditional or unconditional)
            probs_BM: Joint probabilities p(w|x) for all w
            cond_entropy_B: Conditional entropy H(w|x,y)
            uncond_entropy_B: Unconditional entropy H(w|x)
        """
        B, N, D = logits_BND.shape
        device = logits_BND.device

        # Softmax to get probabilities
        probs_BND = F.softmax(logits_BND, dim=-1)

        # Compute log p(w) for all possible w combinations
        # all_w: (M, n_digits) where M = D^n_digits
        M = self.all_w.shape[0]

        # log p(w) = sum_i log p(w_i)
        log_probs_BM = torch.zeros(B, M, device=device)
        for i in range(N):
            log_probs_BM += torch.log(probs_BND[:, i, self.all_w[:, i]] + 1e-10)

        probs_BM = torch.exp(log_probs_BM)  # (B, M)

        # Mask for w's that satisfy constraint: f(w) = y
        # all_y: (M, Y), y_BY: (B, Y)
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

        # Unconditional entropy H(w|x) - entropy over ALL w's
        uncond_entropy_B = -torch.sum(
            probs_BM * torch.log(probs_BM + 1e-10),
            dim=-1
        ) / N  # Normalize by number of concepts

        # Select which entropy to use based on setting
        entropy_B = cond_entropy_B if self.conditional_entropy else uncond_entropy_B

        return loss_B, entropy_B, probs_BM, cond_entropy_B, uncond_entropy_B

    def predict_y_from_probs(self, probs_BM: Tensor) -> Tensor:
        """
        Predict y using argmax over p(y|x) = sum_w p(w|x) * 1{f(w)=y}

        This is the CORRECT way to compute y with high-entropy distributions.
        """
        B = probs_BM.shape[0]
        device = probs_BM.device

        # Get unique y values
        unique_y = torch.unique(self.all_y, dim=0)  # (K, Y)
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
        return self.all_w[best_w_idx]  # (B, n_digits)

    def loss(self, x: Tensor, y: Tensor, log, w_labels: Optional[Tensor] = None) -> Tensor:
        """
        Compute semantic loss with entropy regularization (exact enumeration).

        Loss = -log p(y|x) - lambda * H(w|x) or H(w|x,y)

        Args:
            x: Input images (B, n_digits*28, 28)
            y: Target sum in digit form (B, N+1)
            log: Logger for tracking metrics
            w_labels: Ground truth digits (B, n_digits) - optional, for logging

        Returns:
            loss: Scalar loss value
        """
        self.train()

        # Get digit logits
        logits_BND = self.encode_digits(x)  # (B, n_digits, 10)

        # Compute loss, entropy, and joint probabilities (exact)
        loss_B, entropy_B, probs_BM, cond_ent_B, uncond_ent_B = self.compute_semantic_loss_and_entropy(logits_BND, y)

        # Total loss: semantic loss - entropy regularization
        loss = loss_B.mean() - self.entropy_weight * entropy_B.mean()

        # Logging
        if log is not None:
            log.loss += loss.item()

            with torch.no_grad():
                # Predict y using proper marginalization
                y_pred = self.predict_y_from_probs(probs_BM)
                # Predict w conditioned on predicted y
                w_pred = self.predict_w_from_probs(probs_BM, y_pred)

                # Y accuracy
                y_acc = (y_pred == y).all(dim=-1).float().mean()
                log.var_accuracy_y += y_acc.item()

                # W accuracy
                if w_labels is not None:
                    w_acc = (w_pred == w_labels).float().mean()
                    log.w_acc_train += w_acc.item()
                    log.var_accuracy_w += w_acc.item()

                # Log entropy
                if hasattr(log, 'var_entropy'):
                    log.var_entropy += entropy_B.mean().item()

        return loss

    def evaluate(self, x: Tensor, y: Tensor, w_labels: Tensor, log) -> Dict[str, Tensor]:
        """
        Evaluate the model on a batch using proper joint distribution.

        Args:
            x: Input tensor (B, n_digits*28, 28)
            y: Target sum in digit form (B, N+1)
            w_labels: Ground truth digit labels (B, n_digits)
            log: Logger for tracking metrics

        Returns:
            result_dict: Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            logits_BND = self.encode_digits(x)

            # Compute joint probabilities
            B, N, D = logits_BND.shape
            probs_BND = F.softmax(logits_BND, dim=-1)

            log_probs_BM = torch.zeros(B, self.all_w.shape[0], device=x.device)
            for i in range(N):
                log_probs_BM += torch.log(probs_BND[:, i, self.all_w[:, i]] + 1e-10)
            probs_BM = torch.exp(log_probs_BM)

            # Predict y using argmax p(y|x)
            y_pred = self.predict_y_from_probs(probs_BM)

            # Predict w using argmax p(w|x,y) given predicted y
            w_pred = self.predict_w_from_probs(probs_BM, y_pred)

            # Digit accuracy (concept accuracy)
            digit_acc = (w_pred == w_labels).float().mean()
            log.w_acc_avg += digit_acc.item()

            # Sum accuracy
            sum_acc = (y_pred == y).all(dim=-1).float().mean()
            log.y_acc_avg += sum_acc.item()
            log.y_acc_top += sum_acc.item()

            # Return predictions
            result_dict = {
                "LABELS": y,
                "CONCEPTS": w_labels,
                "w_MM": w_pred,
                "w_TM": w_pred,
                "y_MMf": y_pred,
                "y_TMf": y_pred,
                "y_fMM": y_pred,
                "y_fTM": y_pred,
                "W_SAMPLES": w_pred.unsqueeze(0),
                "Y_SAMPLES": y_pred.unsqueeze(0),
            }

            # Update pred_types in log
            for key in ["w_MM", "w_TM"]:
                if hasattr(log, 'pred_types') and key in log.pred_types:
                    log.pred_types[key] += digit_acc.item()
            for key in ["y_MMf", "y_TMf", "y_fMM", "y_fTM"]:
                if hasattr(log, 'pred_types') and key in log.pred_types:
                    log.pred_types[key] += sum_acc.item()

        return result_dict


# ==========================================
# FFT-Based Scalable Implementation for Large N
# ==========================================

def discrete_convolution_fft(dist1: Tensor, dist2: Tensor) -> Tensor:
    """
    Computes convolution using FFT (Fast Fourier Transform).
    Complexity: O(L log L) instead of O(L^2).
    Essential for N >= 3.

    Args:
        dist1: (B, L1) probability distribution
        dist2: (B, L2) probability distribution

    Returns:
        result: (B, L1+L2-1) convolved distribution
    """
    l1 = dist1.shape[1]
    l2 = dist2.shape[1]
    target_len = l1 + l2 - 1

    # FFT is fastest when length is a power of 2
    fft_len = 1
    while fft_len < target_len:
        fft_len *= 2

    # Convert to Frequency Domain (Real FFT)
    f1 = torch.fft.rfft(dist1, n=fft_len, dim=1)
    f2 = torch.fft.rfft(dist2, n=fft_len, dim=1)

    # Multiply in Frequency Domain
    f_result = f1 * f2

    # Convert back to Time Domain (Inverse Real FFT)
    result = torch.fft.irfft(f_result, n=fft_len, dim=1)

    # Crop to actual target length and ensure positivity
    result = result[:, :target_len]
    return torch.clamp(result, min=0.0)


def scale_distribution(probs: Tensor, scale: int) -> Tensor:
    """
    Scales the support of the distribution by 'scale'.
    If probs = [0.1, 0.9] (support 0, 1) and scale=10,
    Output = [0.1, 0, 0, ..., 0.9] (support 0, 10).

    Args:
        probs: (B, n_classes) probability distribution
        scale: scaling factor (power of 10)

    Returns:
        new_probs: (B, (n_classes-1)*scale + 1) scaled distribution
    """
    if scale == 1:
        return probs

    batch_size, n_classes = probs.shape
    device = probs.device

    # New size: (n_classes - 1) * scale + 1
    # e.g., for digits 0-9, max val is 9. Scaled max is 9*scale
    new_len = (n_classes - 1) * scale + 1
    new_probs = torch.zeros(batch_size, new_len, device=device, dtype=probs.dtype)

    # Scatter probabilities to strided indices: 0, scale, 2*scale, ...
    indices = torch.arange(n_classes, device=device) * scale
    indices = indices.unsqueeze(0).expand(batch_size, -1)

    new_probs.scatter_(1, indices, probs)

    return new_probs


class GenerativeNeSyMNISTAddFFT(nn.Module):
    """
    Generative NeSy model for MNIST Addition using FFT-based exact semantic loss.

    This implementation uses FFT convolution to compute p(y|x) exactly in O(L log L) time,
    making it scalable to large N (tested up to N=15).

    Key ideas:
    1. Each digit contributes to the sum scaled by its power of 10
    2. p(sum = k) is computed by convolving scaled digit distributions
    3. FFT makes convolution O(L log L) instead of O(L²)
    4. Entropy regularization on marginal digit distributions

    For MNIST Addition: y = sum of two N-digit numbers represented by 2N images
    """

    def __init__(self, args):
        super(GenerativeNeSyMNISTAddFFT, self).__init__()
        self.N = args.N
        self.n_digits = 2 * args.N  # Total digit positions (2 operands)
        self.n_values = 10  # Each digit can be 0-9
        self.hidden_size = getattr(args, 'cnn_hidden_size', 256)
        self.args = args

        # Hyperparameters
        self.entropy_weight = getattr(args, 'sl_entropy_weight', 0.1)
        # Use shared head for better convergence (default True)
        self.use_shared_head = getattr(args, 'use_shared_head', True)

        # CNN encoder (shared across digits)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.hidden_size),
            nn.ReLU(),
        )

        # Classification head: shared (better) or separate per digit position
        if self.use_shared_head:
            # Single shared head for all digit positions (better data efficiency)
            self.head = nn.Linear(self.hidden_size, self.n_values)
        else:
            # Separate head for each digit position (original approach)
            self.heads = nn.ModuleList([
                nn.Linear(self.hidden_size, self.n_values) for _ in range(self.n_digits)
            ])

        # Pre-compute the maximum possible sum for this N
        # Max sum = 2 * (10^N - 1) = 2 * 999...9 (N digits)
        self.max_sum = 2 * (10 ** self.N - 1)
        print(f"GenerativeNeSyMNISTAddFFT: N={self.N}, max_sum={self.max_sum}")

    def encode_digits(self, x: Tensor) -> Tensor:
        """
        Encode MNIST images to digit logits.

        Args:
            x: Input tensor of shape (B, n_digits*28, 28) or (B, n_digits, 28, 28)

        Returns:
            logits: (B, n_digits, 10)
        """
        B = x.shape[0]

        # Reshape input to (B, n_digits, 28, 28)
        if len(x.shape) == 3:
            x = x.view(B, self.n_digits, 28, 28)

        # Process each digit through shared CNN encoder
        if self.use_shared_head:
            # Batch all digits together for efficiency with shared head
            x_flat = x.view(B * self.n_digits, 1, 28, 28)  # (B*n_digits, 1, 28, 28)
            h_flat = self.encoder(x_flat)  # (B*n_digits, hidden_size)
            logits_flat = self.head(h_flat)  # (B*n_digits, 10)
            logits = logits_flat.view(B, self.n_digits, 10)  # (B, n_digits, 10)
        else:
            # Process each digit with separate heads
            logits_list = []
            for i in range(self.n_digits):
                x_i = x[:, i:i+1, :, :]  # (B, 1, 28, 28)
                h_i = self.encoder(x_i)  # (B, hidden_size)
                logits_i = self.heads[i](h_i)  # (B, 10)
                logits_list.append(logits_i)
            logits = torch.stack(logits_list, dim=1)  # (B, n_digits, 10)

        return logits

    def compute_sum_distribution(self, digit_probs: Tensor) -> Tensor:
        """
        Compute the probability distribution over sums using FFT convolution.

        Args:
            digit_probs: (B, n_digits, 10) probability distributions for each digit

        Returns:
            sum_dist: (B, max_sum+1) probability distribution over possible sums
        """
        B, seq_len, _ = digit_probs.shape
        device = digit_probs.device

        # Initialize with Dirac delta at 0
        current_sum_dist = torch.zeros(B, 1, device=device, dtype=digit_probs.dtype)
        current_sum_dist[:, 0] = 1.0

        N = self.N

        for i in range(seq_len):
            # Determine power of 10 for this digit position
            # Digits 0..N-1: first number (powers 10^{N-1} down to 10^0)
            # Digits N..2N-1: second number (powers 10^{N-1} down to 10^0)
            if i < N:
                power = 10 ** (N - 1 - i)
            else:
                power = 10 ** (N - 1 - (i - N))

            # Get probs for this digit
            d_probs = digit_probs[:, i, :]  # (B, 10)

            # Scale distribution based on power of 10
            scaled_probs = scale_distribution(d_probs, power)

            # Convolve with current sum distribution
            current_sum_dist = discrete_convolution_fft(current_sum_dist, scaled_probs)

            # Renormalize for numerical stability
            current_sum_dist = current_sum_dist / (current_sum_dist.sum(dim=1, keepdim=True) + 1e-10)

        return current_sum_dist

    def _sum_to_digits(self, sums: Tensor) -> Tensor:
        """Convert integer sums to digit representation."""
        num_result_digits = self.N + 1
        device = sums.device

        result = torch.zeros(*sums.shape, num_result_digits, device=device, dtype=torch.int64)
        for i in range(num_result_digits):
            result[..., i] = (sums // (10 ** (num_result_digits - 1 - i))) % 10

        return result

    def _digits_to_sum(self, digits: Tensor) -> Tensor:
        """Convert digit tensor to integer sum."""
        num_digits = digits.shape[-1]
        device = digits.device
        powers = torch.tensor([10 ** (num_digits - 1 - i) for i in range(num_digits)],
                             device=device, dtype=torch.int64)
        return (digits.to(torch.int64) * powers).sum(dim=-1)

    def loss(self, x: Tensor, y: Tensor, log, w_labels: Optional[Tensor] = None) -> Tensor:
        """
        Compute semantic loss with entropy regularization using FFT convolution.

        Loss = -log p(y|x) - lambda * H(w|x)

        Where p(y|x) is computed exactly via FFT convolution.

        Args:
            x: Input images (B, n_digits*28, 28)
            y: Target sum in digit form (B, N+1)
            log: Logger for tracking metrics
            w_labels: Ground truth digits (B, n_digits) - optional, for logging

        Returns:
            loss: Scalar loss value
        """
        self.train()
        B = x.shape[0]
        device = x.device

        # Get digit logits and probabilities
        logits_BND = self.encode_digits(x)  # (B, n_digits, 10)
        probs_BND = F.softmax(logits_BND, dim=-1)

        # Compute sum distribution via FFT convolution
        sum_dist = self.compute_sum_distribution(probs_BND)  # (B, L)

        # Convert target y from digits to integer
        target_sum = self._digits_to_sum(y)  # (B,)

        # Clamp targets to valid range (safety check)
        max_supported = sum_dist.shape[1] - 1
        safe_targets = torch.clamp(target_sum, max=max_supported)

        # Get probability of target sum: p(y|x)
        target_probs = sum_dist.gather(1, safe_targets.view(-1, 1)).squeeze(-1)  # (B,)

        # Semantic loss: -log p(y|x)
        semantic_loss = -torch.log(target_probs + 1e-10).mean()

        # Entropy regularization: H(w|x) = sum_i H(w_i|x)
        # For factorized distribution, entropy is sum of marginal entropies
        entropy_per_digit = -(probs_BND * torch.log(probs_BND + 1e-10)).sum(dim=-1)  # (B, n_digits)
        entropy = entropy_per_digit.mean()  # Average entropy

        # Total loss
        loss = semantic_loss - self.entropy_weight * entropy

        # Logging
        if log is not None:
            log.loss += loss.item()

            with torch.no_grad():
                # Predict sum using argmax over sum distribution
                pred_sum = sum_dist.argmax(dim=-1)  # (B,)

                # Y accuracy
                y_acc = (pred_sum == target_sum).float().mean()
                log.var_accuracy_y += y_acc.item()

                # W accuracy (using marginal argmax)
                if w_labels is not None:
                    w_pred = probs_BND.argmax(dim=-1)  # (B, n_digits)
                    w_acc = (w_pred == w_labels).float().mean()
                    log.w_acc_train += w_acc.item()
                    log.var_accuracy_w += w_acc.item()

                # Log entropy
                if hasattr(log, 'var_entropy'):
                    log.var_entropy += entropy.item()

        return loss

    def evaluate(self, x: Tensor, y: Tensor, w_labels: Tensor, log) -> Dict[str, Tensor]:
        """
        Evaluate the model on a batch.

        Args:
            x: Input tensor (B, n_digits*28, 28)
            y: Target sum in digit form (B, N+1)
            w_labels: Ground truth digit labels (B, n_digits)
            log: Logger for tracking metrics

        Returns:
            result_dict: Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            logits_BND = self.encode_digits(x)
            probs_BND = F.softmax(logits_BND, dim=-1)

            # Compute sum distribution
            sum_dist = self.compute_sum_distribution(probs_BND)

            # Predict sum using argmax
            pred_sum = sum_dist.argmax(dim=-1)  # (B,)
            target_sum = self._digits_to_sum(y)

            # Predict digits using marginal argmax
            w_pred = probs_BND.argmax(dim=-1)  # (B, n_digits)

            # Digit accuracy (concept accuracy)
            digit_acc = (w_pred == w_labels).float().mean()
            log.w_acc_avg += digit_acc.item()

            # Sum accuracy
            sum_acc = (pred_sum == target_sum).float().mean()
            log.y_acc_avg += sum_acc.item()
            log.y_acc_top += sum_acc.item()

            # Convert predicted sum to digits
            y_pred = self._sum_to_digits(pred_sum)

            # Return predictions
            result_dict = {
                "LABELS": y,
                "CONCEPTS": w_labels,
                "w_MM": w_pred,
                "w_TM": w_pred,
                "y_MMf": y_pred,
                "y_TMf": y_pred,
                "y_fMM": y_pred,
                "y_fTM": y_pred,
                "W_SAMPLES": w_pred.unsqueeze(0),
                "Y_SAMPLES": y_pred.unsqueeze(0),
            }

            # Update pred_types in log
            for key in ["w_MM", "w_TM"]:
                if hasattr(log, 'pred_types') and key in log.pred_types:
                    log.pred_types[key] += digit_acc.item()
            for key in ["y_MMf", "y_TMf", "y_fMM", "y_fTM"]:
                if hasattr(log, 'pred_types') and key in log.pred_types:
                    log.pred_types[key] += sum_acc.item()

        return result_dict
