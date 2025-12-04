from __future__ import annotations
import math
import os
import time

from expressive.util import get_device
from torch.utils.data import DataLoader
import torch
import wandb

from expressive.experiments.mnist_op.absorbing_mnist import (
    MNISTAddProblem,
    create_mnistadd,
    vector_to_base10,
)
from expressive.args import MNISTAbsorbingArguments
from expressive.experiments.mnist_op.data import (
    create_nary_multidigit_operation,
    get_mnist_op_dataloaders,
)

from expressive.methods.logger import (
    TestLog,
    TrainingLog,
    TrainLogger,
    TestLogger,
)

SWEEP = True


def print_stats(stats: dict, prefix: str = ""):
    """Print formatted statistics."""
    if prefix:
        print(f"  [{prefix.upper()}] ", end="")
    else:
        print("  ", end="")

    # Format and print each stat
    formatted = []
    for key, value in stats.items():
        # Remove prefix from key for cleaner output
        clean_key = key.split("/")[-1] if "/" in key else key
        if isinstance(value, float):
            formatted.append(f"{clean_key}: {value:.4f}")
        else:
            formatted.append(f"{clean_key}: {value}")

    print(" | ".join(formatted))


def test(
    val_loader: DataLoader,
    test_logger: TestLog,
    model: MNISTAddProblem,
    device: torch.device,
    verbose: bool = True,
):
    for i, batch in enumerate(val_loader):
        mn_digits, label_digits, label = (
            batch[: 2 * args.N],
            batch[2 * args.N : -1],
            batch[-1],
        )
        x = torch.cat(mn_digits, dim=1)
        model.evaluate(
            x.to(device),
            vector_to_base10(label.to(device), args.N + 1),
            torch.stack(label_digits, dim=-1).to(device),
            test_logger.log,
        )
        if args.DEBUG:
            break
    stats = test_logger.push(len(val_loader))

    if verbose:
        # Print validation/test results
        print_stats(stats, test_logger.log.prefix)


args = MNISTAbsorbingArguments(explicit_bool=True).parse_args()


def main():
    # name = "addition_" + str(args.N)
    run = wandb.init(
        project=f"nesy-diffusion",
        # name=name,
        tags=[],
        config=args.__dict__,
        mode="disabled" if not args.use_wandb else "online",
    )

    device = get_device(args)

    model = create_mnistadd(args).to(device)
    arity = 2
    digits_per_number = args.N
    n_operands = arity * digits_per_number

    bin_op = sum if args.op == "sum" else math.prod if args.op == "product" else None
    op = create_nary_multidigit_operation(arity, bin_op)

    if args.DEBUG:
        # Enable anomaly detection in PyTorch for debugging NaNs
        torch.autograd.set_detect_anomaly(True)

        # Add hooks to check for NaNs in gradients
        def hook(grad):
            if torch.isnan(grad).any():
                print("NaN gradient detected!")
                raise RuntimeError("NaN gradient detected")

        for p in model.parameters():
            if p.requires_grad:
                p.register_hook(hook)

    train_size = 60000 if args.test else 50000
    val_size = 0 if args.test else 10000
    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=int(train_size / n_operands),
        count_val=int(val_size / n_operands),
        count_test=int(10000 / n_operands),
        batch_size=args.batch_size,
        n_operands=n_operands,
        op=op,
        # This shuffle is very weird...
        shuffle=True,
    )

    log_iterations = len(train_loader) // args.log_per_epoch

    train_logger = TrainLogger(log_iterations, TrainingLog, args, print_fn=print_stats)
    val_logger = TestLogger(TestLog, args, "val")

    # Print training configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Model type: {'CNN' if args.use_cnn else 'Diffusion'}")
    print(f"  N (digits per operand): {args.N}")
    print(f"  Operation: {args.op}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print("=" * 60 + "\n")

    optim = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0
    )

    os.makedirs(f"models/{run.id}", exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print("\n" + "=" * 60)
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print("=" * 60)

        model.train()
        start_epoch_time = time.time()

        epoch_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            mn_digits, label, w_labels = batch[: 2 * args.N], batch[-1], batch[2 * args.N : -1]

            x = torch.cat(mn_digits, dim=1).to(device)
            w_labels = torch.stack(w_labels, dim=1).to(device)
            label = vector_to_base10(label.to(device), args.N + 1)
            loss = model.loss(x, label, train_logger.log, w_labels)

            epoch_loss += loss.item()
            num_batches += 1

            loss.backward()
            optim.step()

            train_logger.step()

            if args.DEBUG:
                break

        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        avg_epoch_loss = epoch_loss / num_batches

        # Print epoch training summary
        print(f"\n  [TRAIN] Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.2f}s")

        # Get and print latest training stats
        train_stats = train_logger.get_epoch_stats()
        if train_stats:
            stats_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_stats.items() if isinstance(v, float)])
            print(f"  [TRAIN] {stats_str}")

        # If val not available, don't test during training
        if epoch % args.test_every_epochs == 0:
            if not args.test:
                print("\n  ----- VALIDATION -----")
                test(val_loader, val_logger, model, device)
                test_time = time.time() - end_epoch_time
                print(f"  Validation time: {test_time:.2f}s")

            print(f"\n  Saving model to models/{run.id}/model_{epoch}.pth")
            wandb.save(f"model_{epoch}_{run.id}.pth")
            torch.save(model.state_dict(), f"models/{run.id}/model_{epoch}.pth")


    # Final test
    print("\n" + "=" * 60)
    print("FINAL TESTING")
    print("=" * 60)
    test_logger = TestLogger(TestLog, args, "test")
    test_start_time = time.time()
    test(test_loader, test_logger, model, device)
    print(f"\n  Test time: {time.time() - test_start_time:.2f}s")

    print(f"\n  Saving final model to models/{run.id}/model_{epoch}.pth")
    wandb.save(f"model_{epoch}_{run.id}.pth")
    torch.save(model.state_dict(), f"models/{run.id}/model_{epoch}.pth")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
