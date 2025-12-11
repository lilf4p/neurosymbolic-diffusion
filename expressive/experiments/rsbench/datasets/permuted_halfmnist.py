"""
PermutedHalfMNIST Dataset for Ablation Study.

This dataset tests whether NeSyDM relies on specific visual biases of digits 0 and 1.

The key idea:
- Original HalfMNIST: Uses digits 0-4 with specific visual appearance
- Anchors: (0,0)->0 and (0,1)->1 are fully specified
- Ambiguity: (2,3)->5 but (2,0) and (3,0) are never seen

In PermutedHalfMNIST:
- We use a permutation mapping from logical values to visual digits
- E.g., Logic 0 -> Image 5, Logic 1 -> Image 6, Logic 2 -> Image 7, etc.
- The constraint structure remains the SAME
- If NeSyDM succeeds here too, it doesn't rely on visual biases of 0/1
- If NeSyDM fails here, it exploits the simple visual structure of 0/1

Available permutations (configurable via --digit_permutation):
- "identity": [0,1,2,3,4] -> [0,1,2,3,4] (original HalfMNIST for comparison)
- "shift5": [0,1,2,3,4] -> [5,6,7,8,9] (shift by 5 - harder digits)
- "shuffle": [0,1,2,3,4] -> [7,2,9,4,1] (random shuffle - tests robustness)
- "reverse": [0,1,2,3,4] -> [4,3,2,1,0] (reversed ordering)
"""

import torch
from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.mnist_creation import load_2MNIST
from backbones.addmnist_joint import MNISTPairsEncoder, MNISTPairsDecoder
from backbones.addmnist_single import MNISTNeSyDiffClassifier, MNISTNeSyDiffEncoder, MNISTSingleEncoder
from backbones.addmnist_indep_diff import MNISTIndepDiffClassifier
from backbones.addmnist_suggested import MNISTSuggestedEncoder
from backbones.mnistcnn import EntangledDiffusionClassifier, EntangledDiffusionEncoder, MNISTAdditionCNN
from backbones.disjointmnistcnn import DisjointMNISTAdditionCNN
import numpy as np
from copy import deepcopy


# Define different digit permutations for ablation study
PERMUTATIONS = {
    "identity": [0, 1, 2, 3, 4],      # Original HalfMNIST
    "shift5": [5, 6, 7, 8, 9],        # Shift by 5 - harder digits
    "shuffle": [7, 2, 9, 4, 1],       # Random shuffle
    "reverse": [4, 3, 2, 1, 0],       # Reversed
    "swap01": [1, 0, 2, 3, 4],        # Only swap 0 and 1 (anchors)
    "mid": [3, 4, 5, 6, 7],           # Middle digits
}


class PERMUTEDHALFMNIST(BaseDataset):
    """
    PermutedHalfMNIST: Same logical structure as HalfMNIST but with permuted visual digits.

    Use --digit_permutation to select the permutation:
    - identity: Same as HalfMNIST
    - shift5: Uses digits 5-9 instead of 0-4
    - shuffle: Uses [7,2,9,4,1] as visual digits
    - reverse: Uses [4,3,2,1,0] as visual digits
    """
    NAME = "permutedhalfmnist"
    DATADIR = "data/raw"

    def __init__(self, args):
        super().__init__(args)
        # Get the permutation type from args (default to shift5 for the ablation)
        perm_type = getattr(args, 'digit_permutation', 'shift5')
        if perm_type not in PERMUTATIONS:
            raise ValueError(f"Unknown permutation '{perm_type}'. Choose from: {list(PERMUTATIONS.keys())}")

        # Map from logical value (0-4) to visual digit
        self.visual_digits = PERMUTATIONS[perm_type]
        # Inverse map from visual digit to logical value
        self.visual_to_logical = {v: i for i, v in enumerate(self.visual_digits)}

        print(f"PermutedHalfMNIST using permutation '{perm_type}': {self.visual_digits}")
        print(f"  Logical 0 -> Visual digit {self.visual_digits[0]}")
        print(f"  Logical 1 -> Visual digit {self.visual_digits[1]}")
        print(f"  Logical 2 -> Visual digit {self.visual_digits[2]}")
        print(f"  Logical 3 -> Visual digit {self.visual_digits[3]}")
        print(f"  Logical 4 -> Visual digit {self.visual_digits[4]}")

    def get_data_loaders(self):
        # Load FULL MNIST (10 digits) so we can select any permutation
        dataset_train, dataset_val, dataset_test = load_2MNIST(
            n_digits=10,  # Load all 10 digits
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
            args=self.args
        )

        ood_test = self.get_ood_test(dataset_test)

        dataset_train, dataset_val, dataset_test = self.filtrate(
            dataset_train, dataset_val, dataset_test
        )

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.ood_test = ood_test

        self.train_loader = get_loader(
            dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = get_loader(dataset_val, self.args.batch_size, val_test=True)
        self.test_loader = get_loader(dataset_test, self.args.batch_size, val_test=True)
        self.ood_loader = get_loader(ood_test, self.args.batch_size, val_test=True)

        return self.train_loader, self.val_loader, self.test_loader

    def get_ood_loaders(self):
        return [self.ood_loader]

    def get_backbone(self):
        if not self.args.joint:

            if self.args.backbone == "neural":
                return DisjointMNISTAdditionCNN(n_images=self.get_split()[0]), None

            if self.args.backbone == "suggested":
                return MNISTSuggestedEncoder(c_dim=5), MNISTPairsDecoder(
                    c_dim=10, latent_dim=10
                )

            return MNISTSingleEncoder(c_dim=5), MNISTPairsDecoder(
                c_dim=10, latent_dim=10
            )
        else:
            if self.args.backbone == "neural":
                return MNISTAdditionCNN(), None
            return NotImplementedError("Wrong choice")

    def get_backbone_nesydiff(self):
        if self.args.backbone in ["disentangled", "partialentangled"]:
            return MNISTNeSyDiffEncoder(c_dim=5, n_images=2), MNISTNeSyDiffClassifier(embed_all_images=self.args.backbone == "partialentangled", n_images=2, c_dim=5)
        if self.args.backbone == "fullentangled":
            return EntangledDiffusionEncoder(), EntangledDiffusionClassifier(n_images=2, n_classes=5)
        raise NotImplementedError("Wrong choice")

    def get_backbone_indep_diff(self):
        """Get backbone for independent diffusion model."""
        return MNISTNeSyDiffEncoder(c_dim=5, n_images=2), MNISTIndepDiffClassifier(embed_all_images=True, n_images=2, c_dim=5)

    def get_split(self):
        if self.args.joint:
            return 1, (5, 5)
        else:
            return 2, (5,)

    def get_concept_labels(self):
        return [str(i) for i in range(5)]

    def get_labels(self):
        return [str(i) for i in range(9)]  # sums 0-8 for halfmnist (0-4 + 0-4)

    def get_w_dim(self):
        return (2, 5)

    def get_y_dim(self):
        return (1, 19)

    def y_from_w(self, w_SBW: torch.Tensor) -> torch.Tensor:
        return (w_SBW[..., 0] + w_SBW[..., 1]).unsqueeze(-1)

    def _visual_to_logical(self, visual_concepts):
        """Convert visual digit concepts to logical concepts (0-4)."""
        logical = np.zeros_like(visual_concepts)
        for logical_val, visual_digit in enumerate(self.visual_digits):
            logical[visual_concepts == visual_digit] = logical_val
        return logical

    def filtrate(self, train_dataset, val_dataset, test_dataset):
        """
        Filter the dataset to match the HalfMNIST constraint structure,
        but using the permuted visual digits.

        The logical constraint structure (using logical values 0-4):
        - (0,0), (0,1), (1,0): Anchors
        - (2,3), (3,2), (2,4), (4,2): Ambiguous combinations

        But we filter based on VISUAL digits that map to these logical values.
        """
        # Get the visual digits for our logical structure
        v0, v1, v2, v3, v4 = self.visual_digits

        # Filter training set
        train_c_mask1 = (
            # (0,0) -> (v0, v0)
            ((train_dataset.real_concepts[:, 0] == v0) & (train_dataset.real_concepts[:, 1] == v0))
            # (0,1) -> (v0, v1)
            | ((train_dataset.real_concepts[:, 0] == v0) & (train_dataset.real_concepts[:, 1] == v1))
            # (2,3) -> (v2, v3)
            | ((train_dataset.real_concepts[:, 0] == v2) & (train_dataset.real_concepts[:, 1] == v3))
            # (2,4) -> (v2, v4)
            | ((train_dataset.real_concepts[:, 0] == v2) & (train_dataset.real_concepts[:, 1] == v4))
        )
        train_c_mask2 = (
            # (0,0) -> (v0, v0)
            ((train_dataset.real_concepts[:, 1] == v0) & (train_dataset.real_concepts[:, 0] == v0))
            # (1,0) -> (v1, v0)
            | ((train_dataset.real_concepts[:, 1] == v0) & (train_dataset.real_concepts[:, 0] == v1))
            # (3,2) -> (v3, v2)
            | ((train_dataset.real_concepts[:, 1] == v2) & (train_dataset.real_concepts[:, 0] == v3))
            # (4,2) -> (v4, v2)
            | ((train_dataset.real_concepts[:, 1] == v2) & (train_dataset.real_concepts[:, 0] == v4))
        )
        train_mask = np.logical_or(train_c_mask1, train_c_mask2)

        # Same for validation
        val_c_mask1 = (
            ((val_dataset.real_concepts[:, 0] == v0) & (val_dataset.real_concepts[:, 1] == v0))
            | ((val_dataset.real_concepts[:, 0] == v0) & (val_dataset.real_concepts[:, 1] == v1))
            | ((val_dataset.real_concepts[:, 0] == v2) & (val_dataset.real_concepts[:, 1] == v3))
            | ((val_dataset.real_concepts[:, 0] == v2) & (val_dataset.real_concepts[:, 1] == v4))
        )
        val_c_mask2 = (
            ((val_dataset.real_concepts[:, 1] == v0) & (val_dataset.real_concepts[:, 0] == v0))
            | ((val_dataset.real_concepts[:, 1] == v0) & (val_dataset.real_concepts[:, 0] == v1))
            | ((val_dataset.real_concepts[:, 1] == v2) & (val_dataset.real_concepts[:, 0] == v3))
            | ((val_dataset.real_concepts[:, 1] == v2) & (val_dataset.real_concepts[:, 0] == v4))
        )
        val_mask = np.logical_or(val_c_mask1, val_c_mask2)

        # Same for test
        test_c_mask1 = (
            ((test_dataset.real_concepts[:, 0] == v0) & (test_dataset.real_concepts[:, 1] == v0))
            | ((test_dataset.real_concepts[:, 0] == v0) & (test_dataset.real_concepts[:, 1] == v1))
            | ((test_dataset.real_concepts[:, 0] == v2) & (test_dataset.real_concepts[:, 1] == v3))
            | ((test_dataset.real_concepts[:, 0] == v2) & (test_dataset.real_concepts[:, 1] == v4))
        )
        test_c_mask2 = (
            ((test_dataset.real_concepts[:, 1] == v0) & (test_dataset.real_concepts[:, 0] == v0))
            | ((test_dataset.real_concepts[:, 1] == v0) & (test_dataset.real_concepts[:, 0] == v1))
            | ((test_dataset.real_concepts[:, 1] == v2) & (test_dataset.real_concepts[:, 0] == v3))
            | ((test_dataset.real_concepts[:, 1] == v2) & (test_dataset.real_concepts[:, 0] == v4))
        )
        test_mask = np.logical_or(test_c_mask1, test_c_mask2)

        # Apply masks
        train_dataset.data = train_dataset.data[train_mask]
        val_dataset.data = val_dataset.data[val_mask]
        test_dataset.data = test_dataset.data[test_mask]

        train_dataset.concepts = train_dataset.concepts[train_mask]
        val_dataset.concepts = val_dataset.concepts[val_mask]
        test_dataset.concepts = test_dataset.concepts[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        val_dataset.targets = np.array(val_dataset.targets)[val_mask]
        test_dataset.targets = np.array(test_dataset.targets)[test_mask]

        # CRITICAL: Convert real_concepts from visual digits to logical values (0-4)
        # This is what the model will predict, and it needs to be consistent
        train_dataset.real_concepts = self._visual_to_logical(train_dataset.real_concepts[train_mask])
        val_dataset.real_concepts = self._visual_to_logical(val_dataset.real_concepts[val_mask])
        test_dataset.real_concepts = self._visual_to_logical(test_dataset.real_concepts[test_mask])

        # Also update the concept labels (what the model is trained to predict)
        train_dataset.concepts = self._visual_to_logical(train_dataset.concepts)
        val_dataset.concepts = self._visual_to_logical(val_dataset.concepts)
        test_dataset.concepts = self._visual_to_logical(test_dataset.concepts)

        # Recompute targets (sums) based on logical values
        train_dataset.targets = train_dataset.real_concepts[:, 0] + train_dataset.real_concepts[:, 1]
        val_dataset.targets = val_dataset.real_concepts[:, 0] + val_dataset.real_concepts[:, 1]
        test_dataset.targets = test_dataset.real_concepts[:, 0] + test_dataset.real_concepts[:, 1]

        return train_dataset, val_dataset, test_dataset

    def get_ood_test(self, test_dataset):
        """
        Get OOD test set: all valid digit combinations that are NOT in training.
        Uses the same permutation mapping.
        """
        ood_test = deepcopy(test_dataset)

        v0, v1, v2, v3, v4 = self.visual_digits

        # All samples must use our visual digits
        mask_col0 = np.isin(test_dataset.real_concepts[:, 0], self.visual_digits)
        mask_col1 = np.isin(test_dataset.real_concepts[:, 1], self.visual_digits)

        # Exclude in-distribution combinations (same as training)
        test_c_mask1 = (
            ((test_dataset.real_concepts[:, 0] == v0) & (test_dataset.real_concepts[:, 1] == v0))
            | ((test_dataset.real_concepts[:, 0] == v0) & (test_dataset.real_concepts[:, 1] == v1))
            | ((test_dataset.real_concepts[:, 0] == v2) & (test_dataset.real_concepts[:, 1] == v3))
            | ((test_dataset.real_concepts[:, 0] == v2) & (test_dataset.real_concepts[:, 1] == v4))
        )
        test_c_mask2 = (
            ((test_dataset.real_concepts[:, 1] == v0) & (test_dataset.real_concepts[:, 0] == v0))
            | ((test_dataset.real_concepts[:, 1] == v0) & (test_dataset.real_concepts[:, 0] == v1))
            | ((test_dataset.real_concepts[:, 1] == v2) & (test_dataset.real_concepts[:, 0] == v3))
            | ((test_dataset.real_concepts[:, 1] == v2) & (test_dataset.real_concepts[:, 0] == v4))
        )

        test_mask_in_range = np.logical_and(mask_col0, mask_col1)
        test_mask_value = np.logical_and(~test_c_mask1, ~test_c_mask2)
        test_mask = np.logical_and(test_mask_in_range, test_mask_value)

        ood_test.data = ood_test.data[test_mask]
        ood_test.concepts = ood_test.concepts[test_mask]
        ood_test.targets = np.array(ood_test.targets)[test_mask]
        ood_test.real_concepts = ood_test.real_concepts[test_mask]

        # Convert to logical values
        ood_test.real_concepts = self._visual_to_logical(ood_test.real_concepts)
        ood_test.concepts = self._visual_to_logical(ood_test.concepts)
        ood_test.targets = ood_test.real_concepts[:, 0] + ood_test.real_concepts[:, 1]

        return ood_test

    def print_stats(self):
        print("## PermutedHalfMNIST Statistics ##")
        print(f"Permutation: {self.visual_digits}")
        print("Train samples", len(self.dataset_train.data))
        print("Validation samples", len(self.dataset_val.data))
        print("Test samples", len(self.dataset_test.data))
        print("Test OOD samples", len(self.ood_test.data))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--c_sup", type=int, default=1)
    parser.add_argument("--digit_permutation", type=str, default="shift5")
    parser.add_argument("--task", type=str, default="addition")
    args = parser.parse_args()

    dataset = PERMUTEDHALFMNIST(args)
    dataset.get_data_loaders()
    dataset.print_stats()
