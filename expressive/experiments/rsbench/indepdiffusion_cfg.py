"""
Independent Diffusion with Classifier-Free Guidance (CFG) for RSBench.

This script runs the independent diffusion model with CFG:
- During training: randomly drop labels with probability p_uncond
- During inference: interpolate between conditional and unconditional predictions

Usage:
    python indepdiffusion_cfg.py --dataset halfmnist --backbone partialentangled --use_cfg True --cfg_scale 2.0 ...
"""
import argparse
import os
import time

from expressive.args import RSBenchArguments
from expressive.experiments.rsbench.datasets import get_dataset
from expressive.experiments.rsbench.models.indep_diffusion_cfg import create_indep_diffusion_cfg
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
        # Labels: FSLR. We group F and S together.
        new_labels_BY = torch.zeros(size=(labels_B.shape[0], 3), device=labels_B.device, dtype=labels_B.dtype)

        # Recode forward
        mask_F = labels_B[:, 0] == 1
        new_labels_BY[:, 0][mask_F] = 1
        # Recode stop
        mask_S = labels_B[:, 1] == 1
        new_labels_BY[:, 0][mask_S] = 2
        # Recode neither forward nor stop
        mask_NFNS = ~(mask_F | mask_S)
        new_labels_BY[:, 0][mask_NFNS] = 3

        # L:
        new_labels_BY[:, 1] = labels_B[:, 2]
        # R:
        new_labels_BY[:, 2] = labels_B[:, 3]

        return new_labels_BY
    return labels_B.unsqueeze(1)


def decode_label(labels_BY, args: RSBenchArguments):
    if args.dataset == "boia":
        new_labels_B4 = torch.zeros(size=labels_BY.shape[:-1] + (4,), device=labels_BY.device, dtype=labels_BY.dtype)
        # Recode forward
        new_labels_B4[..., 0][labels_BY[..., 0] == 1] = 1
        # Recode stop
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
        # Convert costs into label indices based on possible cost values
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


if __name__ == "__main__":
    args = RSBenchArguments(explicit_bool=True).parse_args()

    # Print CFG settings
    print(f"=== CFG Configuration ===")
    print(f"  use_cfg: {args.use_cfg}")
    print(f"  cfg_p_uncond: {args.cfg_p_uncond}")
    print(f"  cfg_scale: {args.cfg_scale}")
    print(f"=========================")

    run = wandb.init(
        project=f"nesy-indep-diffusion-cfg-rsbench",
        tags=["independent-diffusion", "cfg"],
        config=args.__dict__,
        mode="offline" if not args.use_wandb or args.DEBUG else "online",
    )

    device = get_device(args)

    dataset = get_dataset(args)
    # Use independent diffusion model with CFG
    model = create_indep_diffusion_cfg(args, dataset).to(device)
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

    for epoch in range(0, args.epochs):
        print(f"Epoch {epoch}")
        if epoch % args.test_every_epochs == 0:
            start_test_time = time.time()
            stats = eval(val_loader, val_logger, model, device, args)
            print(stats)
            test_time = time.time() - start_test_time
            print(f"Test time: {test_time:.2f}s")
            for i, ood_loader in enumerate(ood_loaders):
                ood_stats = eval(ood_loader, ood_loggers[i], model, device, args)
                print(ood_stats)

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

    if args.save_model:
        print(f"Saving model to {run.id}")
        wandb.save(f"model_{epoch}_{run.id}.pth")
        os.makedirs(f"models/{run.id}", exist_ok=True)
        torch.save(model.state_dict(), f"models/{run.id}/model_{epoch}.pth")

    test_logger = TestLogger(clazz, args, "test")
    stats = eval(test_loader, test_logger, model, device, args)
    print(stats)

    for i, ood_loader in enumerate(ood_loaders):
        stats = eval(ood_loader, ood_loggers[i], model, device, args)
        print(stats)
