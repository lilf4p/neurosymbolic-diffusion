# Semantic Loss module
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.normal_kl_divergence import kl_divergence


def rloo_loss(log_probs_SB, reward_SB):
    """
    Reinforce with leave-one-out (RLOO) loss.

    This is the key difference from standard semantic loss:
    - Instead of marginalizing over all worlds, we SAMPLE specific worlds
    - Each sample gets direct feedback (reward) on whether it satisfied constraints
    - The leave-one-out baseline reduces variance in gradient estimates

    Args:
        log_probs_SB: [S, B] log probabilities of sampled concept assignments
        reward_SB: [S, B] binary rewards (1 if constraint satisfied, 0 otherwise)

    Returns:
        loss_B: [B] RLOO loss per batch element

    Mathematical derivation:
    We want to minimize: -log E[1_{c satisfies y}] = -log p(y|x)

    REINFORCE gradient: ∇ = E[log p(c|x) * reward(c)]
    RLOO uses leave-one-out baseline to reduce variance:
        baseline_i = (Σ_{j≠i} reward_j) / (S-1)
        ∇ ≈ (1/S) Σ_i log p(c_i|x) * (reward_i - baseline_i)
    """
    S = log_probs_SB.shape[0]

    # Compute leave-one-out baseline for each sample
    # baseline_i = (sum of all rewards except i) / (S - 1)
    sum_rewards = torch.sum(reward_SB, dim=0, keepdim=True)  # [1, B]
    baseline_SB = (sum_rewards - reward_SB) / (S - 1)  # [S, B]

    # Compute RLOO gradient estimator
    # rho = mean over samples of: log_prob * (reward - baseline)
    advantage_SB = (reward_SB - baseline_SB).detach()  # Stop gradient on advantage
    rho_B = torch.mean(log_probs_SB * advantage_SB, dim=0)  # [B]

    # Mean reward (estimate of p(y|x))
    mu_B = torch.mean(reward_SB, dim=0)  # [B]

    # The loss is designed to estimate -log p(y|x)
    # We use the trick from: https://github.com/ML-KULeuven/catlog/blob/main/ADDITION/addition.py
    # The idea: we want gradient of -log WMC, which is (1/WMC) * d(WMC)/dtheta
    # This formulation achieves that by having interpretable loss = -log(semantic probability)
    epsilon = 1e-8
    loss_B = torch.log((mu_B - rho_B).detach() + rho_B + epsilon)

    return loss_B


class ADDMNIST_SL(torch.nn.Module):
    def __init__(self, loss, logic, args, pcbm=False) -> None:
        super().__init__()
        self.base_loss = loss
        self.logic = logic
        self.pcbm = pcbm
        self.beta = 0.001
        # Whether to use full BCE (positive + negative terms) or just NLL (positive only)
        self.use_full_bce = getattr(args, 'full_bce', False)
        # Whether to use RLOO (Reinforce Leave-One-Out) for direct concept supervision
        self.use_rloo = getattr(args, 'use_rloo', False)
        self.rloo_samples = getattr(args, 'rloo_samples', 16)  # Number of samples for RLOO
        # Weight for entropy regularization (default 0 = disabled)
        self.w_h = getattr(args, 'w_h', 0.0)
        # Type of entropy: 'unconditional' or 'conditional' (like in NeSy diffusion)
        self.entropy_type = getattr(args, 'entropy_type', 'unconditional')
        # Weight for concept consistency loss (like w_denoise in NeSy diffusion)
        self.w_consistency = getattr(args, 'w_consistency', 0.0)
        # Worlds-queries matrix
        if args.task == "addition":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist", "permutedhalfmnist"] else 5
            )
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist", "permutedhalfmnist"] else 5
            )
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.nr_classes = 3

    def compute_query(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_prob: worlds probability

        Returns:
            query prob: query probability
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.logic[:, query]
        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def forward(self, out_dict, args):
        """Forward step of the loss function

        Args:
            self: instance
            out_dict: output dictionary
            args: command line arguments

        Returns:
            loss: semantic loss plus classification loss
            losses: losses dictionary
        """
        loss, losses = self.base_loss(out_dict, args)

        # load from dict
        Y = out_dict["LABELS"]
        pCs = out_dict["pCS"]

        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1, :]
        B = prob_digit1.shape[0]  # Batch size

        if self.use_rloo:
            # ========================================================
            # RLOO: Reinforce with Leave-One-Out baseline
            # ========================================================
            # This approach SAMPLES specific concept assignments and
            # provides direct feedback on whether each sample satisfies
            # the constraint. This is the key difference from standard
            # semantic loss which marginalizes over ALL worlds.
            # ========================================================

            S = self.rloo_samples  # Number of samples

            # Create distributions for sampling
            dist1 = Categorical(probs=prob_digit1)  # [B, n_facts]
            dist2 = Categorical(probs=prob_digit2)  # [B, n_facts]

            # Sample S concept assignments for each batch element
            # c1_SB: [S, B] sampled digit 1 values
            # c2_SB: [S, B] sampled digit 2 values
            c1_SB = dist1.sample((S,))  # [S, B]
            c2_SB = dist2.sample((S,))  # [S, B]

            # Compute log probabilities of sampled concepts
            # log p(c1, c2 | x) = log p(c1|x) + log p(c2|x) (independence)
            log_prob1_SB = dist1.log_prob(c1_SB)  # [S, B]
            log_prob2_SB = dist2.log_prob(c2_SB)  # [S, B]
            log_probs_SB = log_prob1_SB + log_prob2_SB  # [S, B]

            # Compute the predicted label for each sampled concept pair
            # For addition task: y = c1 + c2
            if args.task == "addition":
                y_pred_SB = c1_SB + c2_SB  # [S, B]
            elif args.task == "product":
                y_pred_SB = c1_SB * c2_SB  # [S, B]
            else:
                raise NotImplementedError(f"RLOO not implemented for task {args.task}")

            # Compute rewards: 1 if prediction matches label, 0 otherwise
            Y_expanded = Y.unsqueeze(0).expand(S, -1)  # [S, B]
            reward_SB = (y_pred_SB == Y_expanded).float()  # [S, B]

            # Compute RLOO loss
            sl_B = rloo_loss(log_probs_SB, reward_SB)  # [B]
            sl = sl_B.mean()  # Scalar

            # Log additional metrics for debugging
            mean_reward = reward_SB.mean().item()
            losses.update({
                "sl": sl.item(),
                "sl_type": "rloo",
                "rloo_mean_reward": mean_reward,
                "rloo_samples": S
            })
        else:
            # ========================================================
            # Standard Semantic Loss (marginalization over all worlds)
            # ========================================================

            # Compute worlds probability P(w) (the two digits values are independent)
            Z_1 = prob_digit1[..., None]
            Z_2 = prob_digit2[:, None, :]

            probs = Z_1.multiply(Z_2)

            worlds_prob = probs.reshape(-1, self.n_facts * self.n_facts)

            # Compute query probability P(q)
            query_prob = torch.zeros(
                size=(len(probs), self.nr_classes), device=probs.device
            )

            for i in range(self.nr_classes):
                query = i
                query_prob[:, i] = self.compute_query(query, worlds_prob).view(-1)

            # add a small offset for numerical stability
            eps = 1e-5
            query_prob = query_prob + eps
            with torch.no_grad():
                Z = torch.sum(query_prob, dim=-1, keepdim=True)
            query_prob = query_prob / Z

            if self.use_full_bce:
                # FULL BCE: includes both positive and negative terms
                # Create one-hot targets
                targets = F.one_hot(Y.to(torch.long), num_classes=self.nr_classes).float()

                # Clamp probabilities for numerical stability
                query_prob_clamped = torch.clamp(query_prob, min=eps, max=1-eps)

                # Full BCE loss: -[y * log(p) + (1-y) * log(1-p)]
                # Positive term: -log(p) for correct class (y=1)
                # Negative term: -log(1-p) for incorrect classes (y=0)
                sl = F.binary_cross_entropy(query_prob_clamped, targets, reduction="mean")

                losses.update({"sl": sl.item(), "sl_type": "full_bce"})
            else:
                # Original NLL loss (positive term only)
                sl = F.nll_loss(query_prob.log(), Y.to(torch.long), reduction="mean")
                losses.update({"sl": sl.item(), "sl_type": "nll"})

        if self.pcbm:
            kl_div = 0

            mus = out_dict["MUS"]
            logvars = out_dict["LOGVARS"]
            for i in range(2):
                kl_div += kl_divergence(mus[i], logvars[i])

            loss += self.beta * kl_div
            losses.update({"kl-div": kl_div})

        # ========================================================
        # Entropy Regularization
        # ========================================================
        # Two variants:
        # 1. 'unconditional': Maximize H(c|x) - entropy over all concept values
        # 2. 'conditional': Maximize H(c|x,y) - entropy only over valid solutions
        # The conditional variant is what NeSy diffusion uses.
        # ========================================================
        if self.w_h > 0:
            eps_h = 1e-8

            if self.entropy_type == 'conditional':
                # ========================================================
                # CONDITIONAL ENTROPY H(c|x,y)
                # ========================================================
                # Compute joint probability over all (c1, c2) pairs: P(c1, c2 | x)
                joint_probs = prob_digit1.unsqueeze(2) * prob_digit2.unsqueeze(1)  # [B, n_facts, n_facts]

                # Create mask for valid worlds where constraint is satisfied
                c1_idx = torch.arange(self.n_facts, device=Y.device).unsqueeze(0).unsqueeze(2)
                c2_idx = torch.arange(self.n_facts, device=Y.device).unsqueeze(0).unsqueeze(0)
                if args.task == "addition":
                    predictions = c1_idx + c2_idx
                elif args.task == "product":
                    predictions = c1_idx * c2_idx
                else:
                    predictions = None

                if predictions is not None:
                    Y_expanded = Y.unsqueeze(1).unsqueeze(2)
                    valid_mask = (predictions == Y_expanded).float()

                    # Compute conditional probabilities: P(c|x,y) = P(c|x) * 1_{f(c)=y} / Z
                    masked_probs = joint_probs * valid_mask
                    Z = masked_probs.sum(dim=[1, 2], keepdim=True) + eps_h
                    cond_probs = masked_probs / Z

                    # Compute conditional entropy
                    log_cond_probs = torch.log(cond_probs + eps_h)
                    cond_entropy_terms = cond_probs * log_cond_probs * valid_mask
                    entropy = -cond_entropy_terms.sum(dim=[1, 2]).mean()
                else:
                    # Fallback to unconditional
                    entropy_1 = -torch.sum(prob_digit1 * torch.log(prob_digit1 + eps_h), dim=-1)
                    entropy_2 = -torch.sum(prob_digit2 * torch.log(prob_digit2 + eps_h), dim=-1)
                    entropy = (entropy_1 + entropy_2).mean() / 2
            else:
                # ========================================================
                # UNCONDITIONAL ENTROPY H(c|x)
                # ========================================================
                # Simple factorized entropy over both digits
                entropy_1 = -torch.sum(prob_digit1 * torch.log(prob_digit1 + eps_h), dim=-1)
                entropy_2 = -torch.sum(prob_digit2 * torch.log(prob_digit2 + eps_h), dim=-1)
                entropy = (entropy_1 + entropy_2).mean() / 2

            losses.update({"entropy": entropy.item(), "entropy_type": self.entropy_type})
        else:
            entropy = 0.0

        # ========================================================
        # CONCEPT CONSISTENCY LOSS (like w_denoise in NeSy Diffusion)
        # ========================================================
        # This encourages the model to make consistent predictions
        # on noisy inputs (denoising objective in input space)
        # ========================================================
        consistency_loss = out_dict.get("CONSISTENCY_LOSS", None)
        if consistency_loss is not None and self.w_consistency > 0:
            losses.update({"consistency_loss": consistency_loss.item()})
            consistency_term = self.w_consistency * consistency_loss
        else:
            consistency_term = 0.0

        # ========================================================
        # PROTOTYPE LOSS (for mnistsl_proto)
        # ========================================================
        proto_loss = out_dict.get("PROTO_LOSS", None)
        if proto_loss is not None:
            w_proto = getattr(args, 'w_proto', 1.0)
            losses.update({"proto_loss": proto_loss.item()})
            proto_term = w_proto * proto_loss
        else:
            proto_term = 0.0

        return loss + args.w_sl * sl - self.w_h * entropy + consistency_term + proto_term, losses
