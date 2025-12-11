"""
Ablation Study Script for NeSyDM on HalfMNIST.

This script runs the ablation experiments to test:
1. Visual bias hypothesis: Does NeSyDM rely on specific visual patterns of 0/1?
2. Entropy importance: Is entropy regularization critical for avoiding shortcuts?
3. Denoising importance: Is the w-denoising loss critical?

Experiments:
-----------
1. PERMUTATION ABLATION (Test visual bias):
   - Original HalfMNIST (digits 0-4)
   - Shift5 (digits 5-9) - harder digits with more complex visual patterns
   - Shuffle (random mapping) - tests generality
   - Swap01 (only swap anchor digits 0 and 1)

2. ENTROPY ABLATION:
   - Full model with entropy_weight=1.6
   - No entropy: entropy_weight=0

3. DENOISING ABLATION:
   - Full model with w_denoise_weight=0.0000015
   - No denoising: w_denoise_weight=0

Usage:
------
# Run the full ablation study
python ablation_study.py --experiment all

# Run specific ablation
python ablation_study.py --experiment permutation  # Test visual bias
python ablation_study.py --experiment entropy      # Test entropy importance
python ablation_study.py --experiment denoising    # Test denoising importance

# Run single configuration
python ablation_study.py --dataset permutedhalfmnist --digit_permutation shift5
python ablation_study.py --dataset halfmnist --no_entropy True
"""

import argparse
import os
import time

from expressive.args import RSBenchArguments
from expressive.experiments.rsbench.datasets import get_dataset
from expressive.experiments.rsbench.rsbenchmodel import create_rsbench_diffusion
from expressive.experiments.rsbench.utils.metrics import compute_boia_stats, compute_boia_stats_nesymdm
from expressive.methods.base_model import BaseNeSyDiffusion
from expressive.methods.logger import (
    PRED_TYPES_Y,
    BOIATestLog,
    TestLog,
    TrainingLog,
    TestLogger,
    TrainLogger,
)
from expressive.util import compute_ece_sampled, get_device
from torch.utils.data import DataLoader
import torch
import wandb


def recode_label(labels_B, args: RSBenchArguments):
    if args.dataset == "boia":
        new_labels_BY = torch.zeros(size=(labels_B.shape[0], 3), device=labels_B.device, dtype=labels_B.dtype)
        mask_F = labels_B[:, 0] == 1
        new_labels_BY[:, 0][mask_F] = 1
        mask_S = labels_B[:, 1] == 1
        new_labels_BY[:, 0][mask_S] = 2
        mask_NFNS = ~(mask_F | mask_S)
        new_labels_BY[:, 0][mask_NFNS] = 3
        new_labels_BY[:, 1] = labels_B[:, 2]
        new_labels_BY[:, 2] = labels_B[:, 3]
        return new_labels_BY
    return labels_B.unsqueeze(1)


def decode_label(labels_BY, args: RSBenchArguments):
    if args.dataset == "boia":
        new_labels_B4 = torch.zeros(size=labels_BY.shape[:-1] + (4,), device=labels_BY.device, dtype=labels_BY.dtype)
        new_labels_B4[..., 0][labels_BY[..., 0] == 1] = 1
        new_labels_B4[..., 1][labels_BY[..., 0] == 2] = 1
        new_labels_B4[..., 2:] = labels_BY[..., 1:]
        return new_labels_B4
    return labels_BY.squeeze(1)


def eval(
    val_loader: DataLoader,
    test_logger: TestLog,
    model: BaseNeSyDiffusion,
    device: torch.device,
    args: RSBenchArguments,
):
    print(f"----- {test_logger.prefix} -----")
    print(f"Number of {test_logger.prefix} batches:", len(val_loader))
    master_dict = None
    for i, batch in enumerate(val_loader):
        imgs_BCHW, labels_B, concepts_BW = batch
        concepts_BW = concepts_BW.to(device)
        labels_BY = recode_label(labels_B.to(device), args)
        eval_dict = model.evaluate(
            imgs_BCHW.to(device),
            labels_BY,
            concepts_BW,
            test_logger.log,
        )
        if master_dict is None:
            master_dict = eval_dict
        else:
            master_dict = {k: torch.cat((master_dict[k], eval_dict[k]), dim=-2) for k in eval_dict}

    extra_stats = {}
    extra_stats["ece"] = compute_ece_sampled(master_dict["W_SAMPLES"], master_dict["CONCEPTS"], args.ECE_bins, model.problem.shape_w()[-1])
    if args.dataset == "boia":
        master_dict["LABELS"] = decode_label(master_dict["LABELS"], args)
        master_dict["Y_SAMPLES"] = decode_label(master_dict["Y_SAMPLES"], args)
        for pty in PRED_TYPES_Y:
            master_dict[pty] = decode_label(master_dict[pty], args)
        extra_stats.update(compute_boia_stats_nesymdm(master_dict))
    return test_logger.push(len(val_loader), extra_stats)


def train_and_evaluate(args: RSBenchArguments, experiment_name: str):
    """Train and evaluate a single configuration."""

    # Apply ablation settings
    original_entropy_weight = args.entropy_weight
    original_w_denoise_weight = args.w_denoise_weight

    if args.no_entropy:
        args.entropy_weight = 0.0
        print(f"[ABLATION] Disabling entropy regularization (entropy_weight=0)")

    if args.no_denoising:
        args.w_denoise_weight = 0.0
        print(f"[ABLATION] Disabling w-denoising loss (w_denoise_weight=0)")

    run = wandb.init(
        project=f"nesy-diffusion-ablation",
        tags=["ablation", experiment_name],
        config=args.__dict__,
        mode="offline" if not args.use_wandb or args.DEBUG else "online",
        name=experiment_name,
    )

    device = get_device(args)
    dataset = get_dataset(args)
    model = create_rsbench_diffusion(args, dataset).to(device)
    n_images, c_split = dataset.get_split()

    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    log_iterations = len(train_loader) // args.log_per_epoch
    if log_iterations == 0:
        log_iterations = 1

    train_logger = TrainLogger(log_iterations, TrainingLog, args)
    clazz = BOIATestLog if args.dataset == "boia" else TestLog
    val_logger = TestLogger(clazz, args, "val")

    ood_loaders = dataset.get_ood_loaders()
    ood_loggers = [TestLogger(clazz, args, f"ood_{i + 1}") for i in range(len(ood_loaders))]

    optim = torch.optim.RAdam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0
    )

    best_ood_acc = 0.0
    best_epoch = 0

    for epoch in range(0, args.epochs):
        print(f"Epoch {epoch}")
        if epoch % args.test_every_epochs == 0:
            start_test_time = time.time()
            stats = eval(val_loader, val_logger, model, device, args)
            print(f"Val stats: {stats}")
            test_time = time.time() - start_test_time
            print(f"Test time: {test_time:.2f}s")

            for i, ood_loader in enumerate(ood_loaders):
                ood_stats = eval(ood_loader, ood_loggers[i], model, device, args)
                print(f"OOD stats: {ood_stats}")

                # Track best OOD accuracy
                if "accuracy_w" in ood_stats:
                    if ood_stats["accuracy_w"] > best_ood_acc:
                        best_ood_acc = ood_stats["accuracy_w"]
                        best_epoch = epoch
                        print(f"[NEW BEST] OOD accuracy: {best_ood_acc:.4f} at epoch {epoch}")

        start_epoch_time = time.time()
        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            images, labels, concepts = batch
            labels_BY = recode_label(labels.to(device), args)

            images, labels, concepts = (
                images.to(device),
                labels_BY,
                concepts.to(device),
            )
            loss = model.loss(images, labels.long(), train_logger.log, concepts)
            loss.backward()
            optim.step()

            train_logger.step()

        epoch_time = time.time() - start_epoch_time
        print(f"Epoch time: {epoch_time:.2f}s")

        args.entropy_weight += args.entropy_epoch_increase

    # Final evaluation
    print("\n" + "="*60)
    print(f"FINAL EVALUATION for {experiment_name}")
    print("="*60)

    test_logger = TestLogger(clazz, args, "test")
    test_stats = eval(test_loader, test_logger, model, device, args)
    print(f"Test stats: {test_stats}")

    final_ood_stats = {}
    for i, ood_loader in enumerate(ood_loaders):
        ood_stats = eval(ood_loader, ood_loggers[i], model, device, args)
        print(f"OOD stats: {ood_stats}")
        final_ood_stats[f"ood_{i+1}"] = ood_stats

    # Log summary
    wandb.log({
        "final/best_ood_accuracy": best_ood_acc,
        "final/best_epoch": best_epoch,
        **{f"final/test_{k}": v for k, v in test_stats.items() if isinstance(v, (int, float))},
    })

    if args.save_model:
        os.makedirs(f"models/{run.id}", exist_ok=True)
        torch.save(model.state_dict(), f"models/{run.id}/model_final.pth")

    wandb.finish()

    # Restore original values
    args.entropy_weight = original_entropy_weight
    args.w_denoise_weight = original_w_denoise_weight

    return {
        "experiment": experiment_name,
        "best_ood_accuracy": best_ood_acc,
        "best_epoch": best_epoch,
        "test_stats": test_stats,
        "ood_stats": final_ood_stats,
    }


def run_permutation_ablation(base_args):
    """Run ablation study on different digit permutations."""
    results = []

    permutations = ["identity", "shift5", "shuffle", "reverse", "swap01"]

    for perm in permutations:
        print("\n" + "="*80)
        print(f"RUNNING PERMUTATION ABLATION: {perm}")
        print("="*80 + "\n")

        args = RSBenchArguments(explicit_bool=True).parse_args()
        # Copy base args
        for k, v in base_args.__dict__.items():
            if hasattr(args, k):
                setattr(args, k, v)

        args.dataset = "permutedhalfmnist"
        args.digit_permutation = perm

        result = train_and_evaluate(args, f"permutation_{perm}")
        results.append(result)

    return results


def run_entropy_ablation(base_args):
    """Run ablation study on entropy regularization."""
    results = []

    for no_entropy in [False, True]:
        print("\n" + "="*80)
        print(f"RUNNING ENTROPY ABLATION: no_entropy={no_entropy}")
        print("="*80 + "\n")

        args = RSBenchArguments(explicit_bool=True).parse_args()
        for k, v in base_args.__dict__.items():
            if hasattr(args, k):
                setattr(args, k, v)

        args.no_entropy = no_entropy

        name = "entropy_disabled" if no_entropy else "entropy_enabled"
        result = train_and_evaluate(args, f"entropy_{name}")
        results.append(result)

    return results


def run_denoising_ablation(base_args):
    """Run ablation study on w-denoising loss."""
    results = []

    for no_denoising in [False, True]:
        print("\n" + "="*80)
        print(f"RUNNING DENOISING ABLATION: no_denoising={no_denoising}")
        print("="*80 + "\n")

        args = RSBenchArguments(explicit_bool=True).parse_args()
        for k, v in base_args.__dict__.items():
            if hasattr(args, k):
                setattr(args, k, v)

        args.no_denoising = no_denoising

        name = "denoising_disabled" if no_denoising else "denoising_enabled"
        result = train_and_evaluate(args, f"denoising_{name}")
        results.append(result)

    return results


def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    print(f"\n{'Experiment':<40} {'Best OOD Acc':<15} {'Best Epoch':<10}")
    print("-"*65)

    for r in results:
        print(f"{r['experiment']:<40} {r['best_ood_accuracy']:<15.4f} {r['best_epoch']:<10}")

    print("-"*65)


if __name__ == "__main__":
    # Parse base arguments
    base_args = RSBenchArguments(explicit_bool=True).parse_args()

    # Add experiment selection argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="single",
                        choices=["all", "permutation", "entropy", "denoising", "single"],
                        help="Which ablation experiment to run")
    experiment_args, _ = parser.parse_known_args()

    all_results = []

    if experiment_args.experiment == "all":
        print("Running ALL ablation experiments...")
        all_results.extend(run_permutation_ablation(base_args))
        all_results.extend(run_entropy_ablation(base_args))
        all_results.extend(run_denoising_ablation(base_args))

    elif experiment_args.experiment == "permutation":
        print("Running PERMUTATION ablation...")
        all_results.extend(run_permutation_ablation(base_args))

    elif experiment_args.experiment == "entropy":
        print("Running ENTROPY ablation...")
        all_results.extend(run_entropy_ablation(base_args))

    elif experiment_args.experiment == "denoising":
        print("Running DENOISING ablation...")
        all_results.extend(run_denoising_ablation(base_args))

    else:  # single
        # Run single configuration from command line args
        exp_name = f"{base_args.dataset}"
        if hasattr(base_args, 'digit_permutation') and base_args.dataset == "permutedhalfmnist":
            exp_name += f"_{base_args.digit_permutation}"
        if base_args.no_entropy:
            exp_name += "_noentropy"
        if base_args.no_denoising:
            exp_name += "_nodenoising"

        result = train_and_evaluate(base_args, exp_name)
        all_results.append(result)

    if len(all_results) > 1:
        print_summary(all_results)
