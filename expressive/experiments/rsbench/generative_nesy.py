"""
Generative NeSy Model (without Diffusion)

Key insight from ablation study:
- NeSyDM's entropy term is CRITICAL (removing it → 40% OOD, same as baseline SL)
- NeSyDM's denoising term does NOT influence mode learning
- The key is the GENERATIVE process: sample → evaluate → refine (like RL)

This model captures NeSyDM's essential components WITHOUT diffusion:
1. Sample concepts w from proposal q(w|x)
2. Evaluate constraint satisfaction: reward = 1{f(w) == y}
3. RLOO gradient: refine proposal using leave-one-out baseline
4. Entropy regularization: forces exploration, prevents shortcut collapse

V2: Uses Gumbel-Softmax for differentiable sampling + RLOO for discrete.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.functional import one_hot
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class GenNeSyLog:
    """Training log for Generative NeSy model."""
    accuracy_w: float = 0.0
    accuracy_y: float = 0.0
    avg_reward: float = 0.0
    entropy: float = 0.0
    rloo_loss: float = 0.0
    total_loss: float = 0.0
    n_batches: int = 0

    w_preds: np.ndarray = None
    w_targets: np.ndarray = None

    def __post_init__(self):
        self.w_preds = np.array([])
        self.w_targets = np.array([])

    def reset(self):
        self.accuracy_w = 0.0
        self.accuracy_y = 0.0
        self.avg_reward = 0.0
        self.entropy = 0.0
        self.rloo_loss = 0.0
        self.total_loss = 0.0
        self.n_batches = 0
        self.w_preds = np.array([])
        self.w_targets = np.array([])


class ConceptEncoder(nn.Module):
    """
    Encodes images to concept probability distributions.
    q(w|x) = product of q(w_i|x_i) for each image in the input.
    """

    def __init__(
        self,
        n_images: int,
        c_split: list,
        hidden_dim: int = 256,
        input_channels: int = 1,
    ):
        super().__init__()
        self.n_images = n_images
        self.c_split = c_split  # Number of concepts per image
        self.n_concepts = sum(c_split)

        # Determine n_values from c_split (assume all concepts have same n_values)
        # For MNIST addition: each concept (digit) has 10 possible values (0-9)
        # But for HalfMNIST: each concept has 5 possible values (0-4)
        # We'll set this dynamically based on the task
        self.n_values = None  # Set during forward pass or explicitly

        # Simple CNN encoder for each image
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

        # Output heads for each image's concepts
        # Will be initialized when n_values is known
        self.concept_heads = None
        self._hidden_dim = hidden_dim

    def _init_heads(self, n_values: int, device: torch.device):
        """Initialize concept heads when n_values is known."""
        self.n_values = n_values
        self.concept_heads = nn.ModuleList([
            nn.Linear(self._hidden_dim, n_concepts * n_values)
            for n_concepts in self.c_split
        ]).to(device)

    def forward(self, x_BNX: Tensor, n_values: int = None) -> Tensor:
        """
        Args:
            x_BNX: Input images, shape (B, N, C, H, W) where N is n_images
            n_values: Number of possible values per concept

        Returns:
            logits_BWD: Logits for each concept, shape (B, W, D)
                where W = sum(c_split), D = n_values
        """
        B = x_BNX.shape[0]
        device = x_BNX.device

        # Initialize heads if needed
        if n_values is not None and self.concept_heads is None:
            self._init_heads(n_values, device)

        # Handle different input formats
        if x_BNX.dim() == 4:
            # Shape: (B, N*C, H, W) - need to split into N images
            # Assuming grayscale (C=1) and N=2 images side by side
            if x_BNX.shape[1] == self.n_images:
                # Already (B, N, H, W) - add channel dim
                x_BNX = x_BNX.unsqueeze(2)  # (B, N, 1, H, W)
            else:
                # Assume it's (B, C, H, W) where images are concatenated
                # For 2 MNIST images of 28x28, input could be (B, 1, 28, 56)
                C, H, W = x_BNX.shape[1:]
                if W == 28 * self.n_images:  # Side by side
                    x_BNX = x_BNX.view(B, C, H, self.n_images, 28).permute(0, 3, 1, 2, 4)
                elif H == 28 * self.n_images:  # Stacked vertically
                    x_BNX = x_BNX.view(B, C, self.n_images, 28, W).permute(0, 2, 1, 3, 4)
                else:
                    # Try to reshape as (B, N, C, H, W)
                    x_BNX = x_BNX.view(B, self.n_images, -1, H, W)

        # Encode each image
        all_logits = []
        for i in range(self.n_images):
            # Get i-th image: shape (B, C, H, W)
            x_i = x_BNX[:, i]
            if x_i.dim() == 3:
                x_i = x_i.unsqueeze(1)  # Add channel dim

            # Encode
            h_i = self.encoder(x_i)  # (B, hidden_dim)

            # Get concept logits for this image
            logits_i = self.concept_heads[i](h_i)  # (B, n_concepts_i * n_values)
            logits_i = logits_i.view(B, self.c_split[i], self.n_values)  # (B, n_concepts_i, D)
            all_logits.append(logits_i)

        # Concatenate all concept logits
        logits_BWD = torch.cat(all_logits, dim=1)  # (B, W, D)
        return logits_BWD


class GenerativeNeSy(nn.Module):
    """
    Generative Neurosymbolic Model (without Diffusion).

    Core idea:
    - Learn a proposal distribution q(w|x) over concepts
    - Sample concepts and evaluate constraint satisfaction
    - Use RLOO gradient estimator to refine the proposal
    - Entropy regularization to encourage exploration

    This captures the key insight from NeSyDM:
    - The generative/sampling process is essential
    - The denoising process is NOT essential for mode learning
    """

    def __init__(
        self,
        n_images: int,
        c_split: list,
        n_values: int,
        constraint_fn: callable,
        n_samples: int = 16,
        entropy_weight: float = 1.6,
        hidden_dim: int = 256,
        temperature: float = 1.0,
    ):
        """
        Args:
            n_images: Number of input images
            c_split: List of number of concepts per image
            n_values: Number of possible values per concept (e.g., 10 for MNIST digits)
            constraint_fn: Function that maps concepts to labels: y = f(w)
            n_samples: Number of samples for RLOO estimator
            entropy_weight: Weight for entropy regularization
            hidden_dim: Hidden dimension for encoder
            temperature: Temperature for sampling (higher = more exploration)
        """
        super().__init__()

        self.n_images = n_images
        self.c_split = c_split
        self.n_concepts = sum(c_split)
        self.n_values = n_values
        self.constraint_fn = constraint_fn
        self.n_samples = n_samples
        self.entropy_weight = entropy_weight
        self.temperature = temperature

        # Concept encoder: q(w|x)
        self.encoder = ConceptEncoder(n_images, c_split, hidden_dim)
        self.encoder._init_heads(n_values, torch.device('cpu'))

    def get_proposal_distribution(self, x_BX: Tensor) -> Categorical:
        """Get the proposal distribution q(w|x)."""
        logits_BWD = self.encoder(x_BX, self.n_values)
        # Apply temperature
        logits_BWD = logits_BWD / self.temperature
        return Categorical(logits=logits_BWD)

    def sample_concepts(self, x_BX: Tensor, n_samples: int = None) -> Tuple[Tensor, Categorical]:
        """
        Sample concepts from the proposal distribution.

        Returns:
            w_SBW: Sampled concepts, shape (S, B, W)
            dist: The proposal distribution
        """
        if n_samples is None:
            n_samples = self.n_samples

        dist = self.get_proposal_distribution(x_BX)
        w_SBW = dist.sample((n_samples,))  # (S, B, W)

        return w_SBW, dist

    def compute_reward(self, w_SBW: Tensor, y_BY: Tensor) -> Tensor:
        """
        Compute reward as constraint satisfaction.

        We use a soft reward based on how close the prediction is to the target.
        This provides more gradient signal than hard binary rewards.

        Args:
            w_SBW: Sampled concepts, shape (S, B, W)
            y_BY: Target labels, shape (B, Y)

        Returns:
            reward_SBY: Reward for each sample, shape (S, B, Y)
        """
        S, B, W = w_SBW.shape

        # Compute predicted labels from concepts
        y_pred_SBY = self.constraint_fn(w_SBW.reshape(S * B, W)).view(S, B, -1).float()

        # Hard reward: 1 if prediction matches target
        reward_SBY = (y_pred_SBY == y_BY.unsqueeze(0).float()).float()

        return reward_SBY

    def rloo_loss(
        self,
        log_probs_SB: Tensor,
        reward_SBY: Tensor,
    ) -> Tensor:
        """
        REINFORCE with Leave-One-Out baseline.

        This is the key gradient estimator that makes the generative approach work.

        Args:
            log_probs_SB: Log probability of each sample, shape (S, B)
            reward_SBY: Reward for each sample, shape (S, B, Y)

        Returns:
            loss_BY: RLOO loss, shape (B, Y)
        """
        S = log_probs_SB.shape[0]

        # Compute leave-one-out baseline for each sample
        # baseline[s] = mean(reward[s'] for s' != s)
        reward_sum = reward_SBY.sum(dim=0)  # (B, Y)
        baseline_SBY = (reward_sum.unsqueeze(0) - reward_SBY) / (S - 1)

        # RLOO gradient estimate
        # ∇ log p(w) * (reward - baseline)
        advantage_SBY = reward_SBY - baseline_SBY

        rho_BY = torch.mean(
            log_probs_SB[..., None] * advantage_SBY.detach(),
            dim=0,
        )

        mu_BY = torch.mean(reward_SBY, dim=0)

        # Loss = -log(expected reward) approximated via RLOO
        # This gives an estimate of semantic loss
        epsilon = 1e-8
        loss_BY = torch.log((mu_BY - rho_BY).detach() + rho_BY + epsilon)

        return loss_BY

    def exact_conditional_entropy(self, dist: Categorical, y_BY: Tensor) -> Tensor:
        """
        Compute exact conditional entropy H(w|x, y) using the constraint function.

        This is more informative than unconditional entropy because it only
        encourages diversity among solutions that satisfy the constraint.

        Args:
            dist: Proposal distribution q(w|x)
            y_BY: Target labels, shape (B, Y)

        Returns:
            entropy_B: Conditional entropy, shape (B,)
        """
        # Get probabilities
        probs_BWD = dist.probs  # (B, W, D)
        B, W, D = probs_BWD.shape

        # For computational tractability with small W (e.g., 2 digits),
        # we can enumerate all possible w values
        if W <= 4 and D <= 10:
            # Enumerate all possible concept combinations
            # This is O(D^W) but tractable for small W
            device = probs_BWD.device

            # Create all possible w combinations
            indices = torch.cartesian_prod(*[torch.arange(D, device=device) for _ in range(W)])
            # indices shape: (D^W, W)
            n_combinations = indices.shape[0]

            # Compute probability of each combination under the proposal
            # p(w) = prod_i p(w_i)
            log_probs = torch.zeros(B, n_combinations, device=device)
            for i in range(W):
                # probs_BWD[:, i, :] is (B, D)
                # indices[:, i] is (D^W,)
                log_probs += torch.log(probs_BWD[:, i, indices[:, i]] + 1e-10)

            probs = torch.exp(log_probs)  # (B, D^W)

            # Compute y for each combination
            y_pred = self.constraint_fn(indices.unsqueeze(0).expand(B, -1, -1).reshape(-1, W))
            y_pred = y_pred.view(B, n_combinations, -1)  # (B, D^W, Y)

            # Mask for combinations that satisfy constraint
            mask = (y_pred == y_BY.unsqueeze(1)).all(dim=-1).float()  # (B, D^W)

            # Conditional probability: p(w|y) = p(w) * 1{f(w)=y} / Z
            masked_probs = probs * mask
            Z = masked_probs.sum(dim=-1, keepdim=True) + 1e-10
            conditional_probs = masked_probs / Z  # (B, D^W)

            # Conditional entropy: H(w|y) = -sum p(w|y) log p(w|y)
            entropy_B = -torch.sum(
                conditional_probs * torch.log(conditional_probs + 1e-10),
                dim=-1
            )

            return entropy_B
        else:
            # Fall back to unconditional entropy for large W
            return dist.entropy().sum(dim=-1)

    def loss(
        self,
        x_BX: Tensor,
        y_BY: Tensor,
        log: GenNeSyLog = None,
        eval_w_BW: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the training loss.

        We use a combination of:
        1. Semantic Loss: -log p(y|x) = -log sum_w p(w|x) * 1{f(w)=y}
        2. RLOO for gradient through sampling
        3. Entropy regularization

        Args:
            x_BX: Input images
            y_BY: Target labels
            log: Training log to record stats
            eval_w_BW: Ground truth concepts for evaluation

        Returns:
            loss: Scalar training loss
        """
        self.train()

        # Get distribution over concepts
        dist = self.get_proposal_distribution(x_BX)
        probs_BWD = dist.probs  # (B, W, D)
        B, W, D = probs_BWD.shape
        device = probs_BWD.device

        # ===== Exact Semantic Loss (for small W) =====
        # Enumerate all possible w combinations and compute p(y|x)
        if W <= 4 and D <= 10:
            # Create all possible w combinations
            indices = torch.cartesian_prod(*[torch.arange(D, device=device) for _ in range(W)])
            n_combinations = indices.shape[0]

            # Compute probability of each combination: p(w) = prod_i p(w_i)
            log_probs = torch.zeros(B, n_combinations, device=device)
            for i in range(W):
                log_probs += torch.log(probs_BWD[:, i, indices[:, i]] + 1e-10)
            probs = torch.exp(log_probs)  # (B, D^W)

            # Compute y for each combination
            y_pred = self.constraint_fn(indices.unsqueeze(0).expand(B, -1, -1).reshape(-1, W))
            y_pred = y_pred.view(B, n_combinations, -1)  # (B, D^W, Y)

            # Mask for combinations that satisfy constraint
            mask = (y_pred == y_BY.unsqueeze(1)).all(dim=-1).float()  # (B, D^W)

            # p(y|x) = sum_w p(w|x) * 1{f(w)=y}
            p_y = (probs * mask).sum(dim=-1) + 1e-10  # (B,)

            # Semantic loss = -log p(y|x)
            L_semantic = -torch.log(p_y).mean()

            # Conditional entropy: p(w|x,y) = p(w|x) * 1{f(w)=y} / p(y|x)
            p_w_given_y = (probs * mask) / p_y.unsqueeze(-1)  # (B, D^W)
            entropy_B = -torch.sum(
                p_w_given_y * torch.log(p_w_given_y + 1e-10),
                dim=-1
            )
            entropy = entropy_B.mean()

            # For logging: compute average reward (how often random samples hit target)
            avg_reward = mask.mean().item()
        else:
            # Fall back to RLOO for large W
            w_SBW, _ = self.sample_concepts(x_BX)
            log_probs_SBW = dist.log_prob(w_SBW)
            log_probs_SB = log_probs_SBW.sum(dim=-1)
            reward_SBY = self.compute_reward(w_SBW, y_BY)
            L_rloo_BY = self.rloo_loss(log_probs_SB, reward_SBY)
            L_semantic = L_rloo_BY.mean()
            entropy = dist.entropy().sum(dim=-1).mean()
            avg_reward = reward_SBY.mean().item()

        # Total loss: semantic loss - entropy regularization
        loss = L_semantic - self.entropy_weight * entropy

        # Log statistics
        if log is not None:
            log.n_batches += 1
            log.rloo_loss += L_semantic.item()
            log.entropy += entropy.item()
            log.total_loss += loss.item()
            log.avg_reward += avg_reward

            # Accuracy on most likely prediction
            w_pred_BW = dist.probs.argmax(dim=-1)
            y_pred_BY = self.constraint_fn(w_pred_BW)
            log.accuracy_y += (y_pred_BY == y_BY).float().mean().item()

            if eval_w_BW is not None:
                log.accuracy_w += (w_pred_BW == eval_w_BW).float().mean().item()
                log.w_preds = np.concatenate([log.w_preds, w_pred_BW.flatten().detach().cpu().int().numpy()])
                log.w_targets = np.concatenate([log.w_targets, eval_w_BW.flatten().detach().cpu().int().numpy()])

        return loss

    @torch.no_grad()
    def predict(self, x_BX: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict concepts and labels.

        Returns:
            w_BW: Predicted concepts (argmax)
            y_BY: Predicted labels
        """
        self.eval()
        dist = self.get_proposal_distribution(x_BX)
        w_BW = dist.probs.argmax(dim=-1)
        y_BY = self.constraint_fn(w_BW)
        return w_BW, y_BY

    @torch.no_grad()
    def evaluate(
        self,
        x_BX: Tensor,
        y_BY: Tensor,
        w_BW: Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate model on a batch.

        Returns dictionary with accuracy metrics.
        """
        self.eval()

        w_pred_BW, y_pred_BY = self.predict(x_BX)

        return {
            'accuracy_y': (y_pred_BY == y_BY).float().mean().item(),
            'accuracy_w': (w_pred_BW == w_BW).float().mean().item(),
            'accuracy_w_per_digit': [(w_pred_BW[:, i] == w_BW[:, i]).float().mean().item()
                                      for i in range(w_BW.shape[-1])],
        }


def create_addition_constraint(n_values: int = 5):
    """
    Create constraint function for addition task.

    For HalfMNIST: n_values = 5 (digits 0-4, sums 0-8)
    For full MNIST: n_values = 10 (digits 0-9, sums 0-18)

    Args:
        n_values: Number of possible digit values

    Returns:
        constraint_fn: Function that maps (B, 2) -> (B, 1) as sum
    """
    def constraint_fn(w_BW: Tensor) -> Tensor:
        # w_BW: (B, 2) - two digits
        # Returns: (B, 1) - sum
        return w_BW.sum(dim=-1, keepdim=True)

    return constraint_fn


def create_generative_nesy_for_halfmnist(
    n_samples: int = 16,
    entropy_weight: float = 1.6,
    hidden_dim: int = 256,
) -> GenerativeNeSy:
    """
    Create a GenerativeNeSy model for the HalfMNIST addition task.

    Args:
        n_samples: Number of samples for RLOO
        entropy_weight: Weight for entropy regularization
        hidden_dim: Hidden dimension for encoder

    Returns:
        model: GenerativeNeSy model
    """
    return GenerativeNeSy(
        n_images=2,
        c_split=[1, 1],  # One concept (digit) per image
        n_values=5,  # HalfMNIST uses digits 0-4
        constraint_fn=create_addition_constraint(n_values=5),
        n_samples=n_samples,
        entropy_weight=entropy_weight,
        hidden_dim=hidden_dim,
    )


if __name__ == "__main__":
    # Quick test
    model = create_generative_nesy_for_halfmnist()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    B = 4
    x = torch.randn(B, 2, 1, 28, 28)  # 2 MNIST images
    y = torch.randint(0, 9, (B, 1))  # Sum labels
    w = torch.randint(0, 5, (B, 2))  # Ground truth concepts

    log = GenNeSyLog()
    loss = model.loss(x, y, log, w)
    print(f"Loss: {loss.item():.4f}")
    print(f"Entropy: {log.entropy:.4f}")
    print(f"RLOO loss: {log.rloo_loss:.4f}")

    w_pred, y_pred = model.predict(x)
    print(f"Predicted concepts shape: {w_pred.shape}")
    print(f"Predicted labels shape: {y_pred.shape}")
