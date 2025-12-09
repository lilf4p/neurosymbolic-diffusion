# Classifier-Free Guidance for NeSy Diffusion
#
# This module extends the independent diffusion approach with classifier-free guidance (CFG).
#
# Key differences from standard NeSy diffusion:
# 1. During training: randomly drop the label y with probability p_uncond
#    - This trains the model to predict p(c|x) unconditionally
# 2. During inference: interpolate between conditional and unconditional predictions
#    - p_guided(c|x,y) = p(c|x) + w * (p(c|x,y) - p(c|x))
#    - where w is the guidance scale
#
# CFG provides stronger label conditioning without a separate classifier.

from typing import Optional, Tuple

import numpy as np

from expressive.methods.base_model import BaseNeSyDiffusion, Problem
from expressive.util import safe_sample_categorical
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch.distributions import Categorical

from expressive.methods.logger import TrainingLog
from expressive.methods.simple_nesy_diff import SimpleNeSyDiffusion


class CFGNeSyDiffusion(SimpleNeSyDiffusion):
    """
    NeSy Diffusion with Classifier-Free Guidance.

    Training:
    - With probability p_uncond, train without constraint (unconditional)
    - Otherwise, train normally with constraint

    Inference:
    - Compute conditional prediction p(c|x,y) via rejection sampling
    - Compute unconditional prediction p(c|x) without rejection
    - Interpolate: p_guided = p_uncond + w * (p_cond - p_uncond)
    """

    def __init__(self, p, problem, args):
        super().__init__(p, problem, args)

        # Probability of unconditional training (label dropout)
        self.p_uncond = getattr(args, 'cfg_p_uncond', 0.1)

        # Guidance scale for inference
        self.cfg_scale = getattr(args, 'cfg_scale', 2.0)

        # Whether to use CFG during inference
        self.use_cfg = getattr(args, 'use_cfg', True)

    def loss(
        self, x_BX: Tensor, y_0_BY: Tensor, log: TrainingLog, eval_w_0_BW: Optional[Tensor] = None
    ) -> Tensor:
        """
        Training loss with label dropout for CFG.

        With probability p_uncond, we skip the constraint check during sampling,
        effectively training the model to predict p(c|x) unconditionally.
        """
        self.train()

        # Initialize embedding of x
        encoding_BWE = self.p.encode_x(x_BX)

        # Create mask matrix
        s_w = y_0_BY.shape[:-1] + self.problem.shape_w()
        bm_BW = torch.ones(s_w[:-1], device=x_BX.device, dtype=torch.long) * s_w[-1]

        # Initialize q(w_0|x, y_0)
        q_w_0_BWD = self.p.distribution(
            bm_BW,
            encoding_BWE,
            torch.zeros_like(y_0_BY[..., 0], device=bm_BW.device),
        )

        # ================================================================
        # CFG: Label dropout during training
        # ================================================================
        # Sample a mask for which batch elements to train unconditionally
        uncond_mask = torch.rand(x_BX.shape[0], device=x_BX.device) < self.p_uncond

        # For unconditional training, we use a dummy label (all masked)
        y_0_BY_masked = y_0_BY.clone()
        # Set masked samples to use a null/masked label
        # We use the mask dimension as the "null" label
        null_label = torch.full_like(y_0_BY, self.mask_dim_y())
        y_0_BY_for_sampling = torch.where(
            uncond_mask.unsqueeze(-1).expand_as(y_0_BY),
            null_label,
            y_0_BY
        )
        # ================================================================

        # Sample w_0 (with or without constraint depending on uncond_mask)
        w_1_BW = torch.full(
            (x_BX.shape[0],) + self.problem.shape_w()[:-1],
            self.mask_dim_w(),
            device=x_BX.device,
        )

        # For conditional samples, use normal rejection sampling
        # For unconditional samples, skip rejection (accept all)
        var_w_0_BW = self.sample_with_cfg_training(
            x_BX,
            w_1_BW,
            y_0_BY,
            y_0_BY_for_sampling,
            uncond_mask,
            encoding_BWE,
        )

        var_w_0_BWD = one_hot(var_w_0_BW, s_w[-1] + 1).float()

        # Sample timesteps
        t_B = torch.rand((x_BX.shape[0],), device=x_BX.device)

        # Compute q(w_t | w_0)
        q_w_t_BWD: Tensor = self.q_w.t_step(var_w_0_BWD, t_B)
        w_t_BW = safe_sample_categorical(Categorical(probs=q_w_t_BWD))

        # Compute p(\tilde{w}_0|w_t, x)
        p_w_0_BWD = self.p.distribution(w_t_BW, encoding_BWE, t_B)[..., :-1]

        # Sample S values for \tilde{w}_0
        tw_0 = Categorical(probs=p_w_0_BWD)
        tw_0_SBW = safe_sample_categorical(tw_0, (self.args.loss_S,))

        # Compute deterministic function for tw_0
        ty_0_SBY = self.problem.y_from_w(tw_0_SBW)

        #####################
        # LOSS FUNCTIONS
        #####################

        # w RLOO denoising loss
        L_w_denoising = self.loss_weight(tw_0.log_prob(var_w_0_BW).mean(-1), t_B).mean()

        # y RLOO denoising loss
        log_probs_SB = tw_0.log_prob(tw_0_SBW).sum(-1)

        # For unconditional training, we don't penalize constraint violations
        # We modify the reward: unconditional samples always get reward 1
        Y_expanded = y_0_BY.unsqueeze(0).expand(tw_0_SBW.shape[0], -1, -1)
        constraint_y0_SBY = (Y_expanded == ty_0_SBY).float()
        reward_y_0_SBY = constraint_y0_SBY

        # For unconditional samples, set reward to 1 (no constraint)
        uncond_mask_expanded = uncond_mask.unsqueeze(0).unsqueeze(-1).expand_as(reward_y_0_SBY)
        reward_y_0_SBY = torch.where(
            uncond_mask_expanded,
            torch.ones_like(reward_y_0_SBY),
            reward_y_0_SBY
        )

        # Compute RLOO loss
        L_denoising_BY = self.rloo_loss(log_probs_SB, reward_y_0_SBY)
        L_y_denoising = self.loss_weight(L_denoising_BY.mean(-1), t_B).mean()

        # Entropy regularization
        q_entropy: Tensor = self.entropy_loss(y_0_BY, q_w_0_BWD).mean()

        # Optional denoising entropy
        entropy_denoising_B = self.entropy_loss(y_0_BY, p_w_0_BWD)
        L_entropy_denoising = self.loss_weight(entropy_denoising_B, t_B).mean() if self.args.denoising_entropy else 0.0

        # Logging
        var_y_0_BY = self.problem.y_from_w(var_w_0_BW)
        var_violations_y_0_BY = var_y_0_BY != y_0_BY

        if eval_w_0_BW is not None:
            log.var_accuracy_w += (var_w_0_BW == eval_w_0_BW).float().mean().item()
            log.w_preds = np.concatenate([log.w_preds, var_w_0_BW.flatten().detach().cpu().int().numpy()])
            log.w_targets = np.concatenate([log.w_targets, eval_w_0_BW.flatten().detach().cpu().int().numpy()])

        log.var_entropy += q_entropy.item()
        log.unmasking_entropy += entropy_denoising_B.mean().item()
        log.w_denoise += L_w_denoising.item()
        log.y_denoise += L_y_denoising.item()
        log.avg_constraints += constraint_y0_SBY.float().mean().item()
        log.avg_var_violations += var_violations_y_0_BY.float().mean().item()
        log.var_accuracy_y += torch.min(~var_violations_y_0_BY, dim=-1)[0].float().mean().item()

        return (
            L_y_denoising
            + self.args.w_denoise_weight * L_w_denoising
            - self.args.entropy_weight * q_entropy
            + self.args.entropy_weight * L_entropy_denoising
        )

    def sample_with_cfg_training(
        self,
        x_BX: Tensor,
        w_1_BW: Tensor,
        y_0_BY: Tensor,
        y_0_BY_for_sampling: Tensor,
        uncond_mask: Tensor,
        encoding_BWE: Tensor,
    ) -> Tensor:
        """
        Sample concepts for CFG training.

        - Conditional samples: use rejection sampling with constraint
        - Unconditional samples: sample without constraint checking
        """
        # Sample all conditionally first (with constraint)
        var_w_0_BW_cond = self.sample(
            x_BX,
            w_1_BW,
            y_0_BY,
            self.args.variational_K,
            self.args.variational_T,
            self.args.variational_K,
            encoding_BWE,
            only_w=True,
        )[0]

        # For unconditional samples, we just sample from the prior without rejection
        # This is done by using the "only_w=True" mode but skipping constraint check
        # We achieve this by sampling directly from the initial distribution

        # Get initial distribution p(w|x) at t=0
        bm_BW = torch.ones_like(w_1_BW) * self.mask_dim_w()
        p_w_0_BWD = self.p.distribution(
            bm_BW,
            encoding_BWE,
            torch.zeros(x_BX.shape[0], device=x_BX.device),
        )[..., :-1]  # Remove mask dimension

        # Sample unconditionally
        var_w_0_BW_uncond = safe_sample_categorical(Categorical(probs=p_w_0_BWD))

        # Combine based on mask
        var_w_0_BW = torch.where(
            uncond_mask.unsqueeze(-1).expand_as(var_w_0_BW_cond),
            var_w_0_BW_uncond,
            var_w_0_BW_cond
        )

        return var_w_0_BW

    def sample_with_cfg(
        self,
        x_BX: Tensor,
        w_T_BW: Tensor,
        y_T_BY: Optional[Tensor],
        num_samples: int,
        T: Optional[int],
        S: int,
        encoding_BWE: Optional[Tensor] = None,
        only_w: bool = False,
        cfg_scale: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample with classifier-free guidance.

        1. Get conditional distribution p(c|x,y) via rejection sampling
        2. Get unconditional distribution p(c|x) without rejection
        3. Interpolate: p_guided = p_uncond + cfg_scale * (p_cond - p_uncond)
        """
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        if not self.use_cfg or cfg_scale == 0:
            # Standard sampling without CFG
            return super().sample(
                x_BX, w_T_BW, y_T_BY, num_samples, T, S, encoding_BWE, only_w
            )

        # For CFG, we modify the distribution function during sampling
        # Store original distribution method
        original_distribution = self.p.distribution

        def cfg_distribution(wy_t, x_encoding, t):
            """
            CFG-modified distribution.

            Computes: p_guided = p_uncond + cfg_scale * (p_cond - p_uncond)
                    = (1 - cfg_scale) * p_uncond + cfg_scale * p_cond

            For cfg_scale > 1, this amplifies the conditional signal.
            """
            # Get conditional distribution (with carry-over unmasking)
            p_cond = original_distribution(wy_t, x_encoding, t)

            # Get unconditional distribution (pretend all concepts are masked)
            w_t = wy_t if isinstance(wy_t, Tensor) else wy_t[0]
            bm = torch.ones_like(w_t) * self.mask_dim_w()
            p_uncond = original_distribution(bm, x_encoding, t)

            # CFG interpolation in logit space for better numerical stability
            # log p_guided = log p_uncond + cfg_scale * (log p_cond - log p_uncond)
            eps = 1e-8
            log_p_cond = torch.log(p_cond + eps)
            log_p_uncond = torch.log(p_uncond + eps)

            log_p_guided = log_p_uncond + cfg_scale * (log_p_cond - log_p_uncond)

            # Convert back to probabilities
            p_guided = torch.softmax(log_p_guided, dim=-1)

            # Preserve mask dimension (last dim should be 0)
            p_guided[..., -1] = 0

            # Re-normalize
            p_guided = p_guided / p_guided.sum(dim=-1, keepdim=True)

            return p_guided

        # Temporarily replace distribution function
        self.p.distribution = cfg_distribution

        try:
            result = super().sample(
                x_BX, w_T_BW, y_T_BY, num_samples, T, S, encoding_BWE, only_w
            )
        finally:
            # Restore original distribution function
            self.p.distribution = original_distribution

        return result


def create_cfg_indep_diffusion(args, dataset):
    """Create an independent diffusion model with CFG support."""
    from expressive.experiments.rsbench.models.indep_diffusion import (
        RSBenchIndepDiffModel, RSBenchIndepAdapter
    )

    model = RSBenchIndepDiffModel(args, dataset)
    problem = RSBenchIndepAdapter(args, dataset)

    return CFGNeSyDiffusion(model, problem, args)
