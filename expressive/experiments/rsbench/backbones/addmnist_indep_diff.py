"""
Independent Diffusion backbone for MNIST.

This uses the same architecture as NeSyDM but enforces independence:
- p(c|x) = prod_i p(c_i|x) instead of joint modeling

The key difference from NeSyDM is that when predicting concept i,
we ONLY condition on concept i's diffusion state, NOT on other concepts' states.
This tests whether the cross-concept conditioning (joint modeling) is what
prevents reasoning shortcuts, not the independence assumption itself.
"""

import torch
import torch.nn as nn

from .addmnist_single import MNISTNeSyDiffEncoder


class MNISTIndepDiffClassifier(nn.Module):
    """
    Independent classifier - each concept predicted from its OWN diffusion state only.

    Unlike MNISTNeSyDiffClassifier which conditions on ALL diffusion states:
        p(c_1|x, w_1, w_2) and p(c_2|x, w_1, w_2)

    This enforces independence by only using the relevant concept's state:
        p(c_1|x, w_1) and p(c_2|x, w_2)
    """
    def __init__(
        self, n_images, embed_all_images=False, hidden_channels=32, c_dim=10, latent_dim=16, dropout=0.5
    ):
        super(MNISTIndepDiffClassifier, self).__init__()
        assert n_images == 2, "Only 2 images are supported for now"

        self.hidden_channels = hidden_channels
        self.c_dim = c_dim
        self.latent_dim = latent_dim
        self.n_images = n_images

        self.unflatten_dim = (3, 7)
        self.embed_all_images = embed_all_images

        # Input: encoding for ONE image + ONE diffusion state (c_dim + 1 for the state)
        # Per-image encoding size
        per_image_encoding_size = int(
            4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1] * (3 / 7)
        )

        # Each classifier head only takes: its image encoding + its diffusion state
        self.dense_c = nn.Linear(
            in_features=per_image_encoding_size + (self.c_dim + 1),  # Only ONE concept's state
            out_features=self.c_dim,
        )

    def forward(self, x_encodings, w_0_BWD, image_to_classify: int = -1):
        """
        Predict concept with INDEPENDENCE - each concept only sees its own diffusion state.

        Args:
            x_encodings: Image encodings [B, D] concatenated for all images
            w_0_BWD: Diffusion states [B, W, D] where W is number of concepts
            image_to_classify: Which image/concept to classify (-1 = all)
        """
        w_0_1_BD = w_0_BWD[..., 0, :]  # Diffusion state for concept 1
        w_0_2_BD = w_0_BWD[..., 1, :]  # Diffusion state for concept 2

        # Split encodings per image
        encoding_split = torch.split(x_encodings, x_encodings.size(-1) // self.n_images, dim=-1)
        x1_enc = encoding_split[0]
        x2_enc = encoding_split[1]

        if self.embed_all_images:
            # INDEPENDENT: Each concept only uses ITS OWN diffusion state
            # p(c_1|x_1, w_1) - NOT conditioned on w_2!
            c1 = self.dense_c(torch.cat((w_0_1_BD, x1_enc), dim=-1))
            # p(c_2|x_2, w_2) - NOT conditioned on w_1!
            c2 = self.dense_c(torch.cat((w_0_2_BD, x2_enc), dim=-1))
            return torch.stack([c1, c2], dim=-2)

        if image_to_classify == 0:
            return self.dense_c(torch.cat((w_0_1_BD, x1_enc), dim=-1))
        elif image_to_classify == 1:
            return self.dense_c(torch.cat((w_0_2_BD, x2_enc), dim=-1))
        raise ValueError(f"Invalid image to classify: {image_to_classify}")
