from abc import ABC, abstractmethod

import numpy as np
import wandb

from typing import Type, Generic, TypeVar

from expressive.args import AbsArguments, Arguments

# Add other types to also log them.
PRED_TYPES_W = ["w_TM"]
PRED_TYPES_Y = ["y_fTM"]

class Log(ABC):
    # Should have no arguments in __init__!
    # Just create defaults (eg set loss = 0) and then update manually
    def __init__(self, args: Arguments):
        pass

    @abstractmethod
    def create_dict(self, iterations: int) -> dict:
        pass


LOG = TypeVar("LOG", bound=Log)
GLOBAL_ITERATIONS = 0


class TrainingLog(Log):
    def __init__(self, args: AbsArguments):
        self.var_entropy = 0.0
        self.unmasking_entropy = 0.0
        self.w_denoise = 0.0
        self.y_denoise = 0.0
        self.Z_loss = 0.0
        self.avg_violation = 0.0
        self.avg_constraints = 0.0
        self.avg_var_violations = 0.0
        self.var_accuracy_y = 0.0
        self.var_accuracy_w = 0.0

        # For simple CNN model
        self.loss = 0.0
        self.w_acc_train = 0.0

        self.w_preds = np.array([], dtype=np.int32)
        self.w_targets = np.array([], dtype=np.int32)
        self.args = args

    def create_dict(self, iterations: int) -> dict:
        def norm(x):
            return x / iterations

        # Check if this is a simple CNN model training (loss is set)
        if self.loss > 0:
            # Simple CNN model logging
            return {
                "loss": norm(self.loss),
                "w_acc_train": norm(self.w_acc_train),
            }

        var_entropy_norm = norm(self.var_entropy)
        unmasking_entropy_norm = norm(self.unmasking_entropy)
        w_denoise_norm = norm(self.w_denoise)
        y_denoise_norm = norm(self.y_denoise)
        log_z_norm = norm(self.Z_loss)
        violations_norm = norm(self.avg_violation)
        constraints = norm(self.avg_constraints)
        var_violations_norm = norm(self.avg_var_violations)
        var_accuracy_y = norm(self.var_accuracy_y)
        var_accuracy_w = norm(self.var_accuracy_w)
        # Entropy and log z are negated in the loss
        loss = -var_entropy_norm + w_denoise_norm + y_denoise_norm - log_z_norm

        base_dict =  {
            "var_entropy": var_entropy_norm,
            "unmasking_entropy": unmasking_entropy_norm,
            "w_denoise": w_denoise_norm,
            "y_denoise": y_denoise_norm,
            "loss": loss,
            "avg_constraints": constraints,
            "avg_var_violations": var_violations_norm,
            "var_accuracy_y": var_accuracy_y,
            "var_accuracy_w": var_accuracy_w,
        }

        if self.args.send_conf_matrix:
            base_dict["conf_matrix_w"] = wandb.plot.confusion_matrix(
                probs=None,
                y_true=self.w_targets,
                preds=self.w_preds,
                # class_names=["0.8", "1.2", "5.3", "7.7", "9.2"],
            ),

        if not self.args.simple_model:
            base_dict["log_z"] = log_z_norm
            base_dict["avg_violation"] = violations_norm
        return base_dict


class TestLog(Log):
    def __init__(self, args: Arguments, prefix: str):
        self.y_acc_avg = 0.0
        self.w_acc_avg = 0.0
        self.w_acc_top = 0.0
        self.y_acc_top = 0.0
        self.pred_types = {
            ptw: 0.0 for ptw in PRED_TYPES_W
        }
        self.pred_types.update({
            pty: 0.0 for pty in PRED_TYPES_Y
        })
        self.args = args
        self.prefix = prefix

    def create_dict(self, iterations: int) -> dict:
        def norm(x):
            return x / iterations

        base_dict = {
            "w_acc_avg": norm(self.w_acc_avg),
            "w_acc_top": norm(self.w_acc_top),
        }
        for key, value in self.pred_types.items():
            base_dict[key] = norm(value)
        if not self.args.simple_model:
            base_dict["y_acc_avg"] = norm(self.y_acc_avg)
            base_dict["y_acc_top"] = norm(self.y_acc_top)
        return {
            self.prefix + "/" + key: value
            for key, value in base_dict.items()
        }

class BOIATestLog(TestLog):
    def __init__(self, args: Arguments, prefix: str):
        super().__init__(args, prefix)
        self.w_preds = {
            ptw: np.empty((0, 21), dtype=np.int32) for ptw in PRED_TYPES_W
        }
        self.y_preds = {
            pty: np.empty((0, 3), dtype=np.int32) for pty in PRED_TYPES_Y
        }
        self.w_targets_B21 = np.empty((0, 21), dtype=np.int32)
        self.y_targets_B3 = np.empty((0, 3), dtype=np.int32)

    def create_dict(self, iterations: int) -> dict:
        stats_dict = {}
        if self.args.send_conf_matrix:
            def conf_matrix_y(index: int, name: str, class_names: list[str]):
                stats_dict[f"conf_matrix_{name}"] = wandb.plot.confusion_matrix(
                    probs=None,
                    title=name,
                    y_true=self.y_targets_B3[:, index],
                    preds=self.y_preds_B3[:, index],
                    class_names=class_names
                )
            conf_matrix_y(0, "FS", ["invalid", "forward", "stop", "neither"])
            conf_matrix_y(1, "L", ["no left", "left turn"])
            conf_matrix_y(2, "R", ["no right", "right turn"])
            def conf_matrix_w(index: int, name: str):
                stats_dict[f"conf_matrix_{name}"] = wandb.plot.confusion_matrix(
                    probs=None,
                    title=name,
                    y_true=self.w_targets_B21[:, index],
                    preds=self.w_preds_B21[:, index],
                    class_names=["False", "True"]
                )
            names = [
                "TrafficLightGreenFS",
                "FollowCarFS",
                "ClearFS",
                "TrafficLightRedFS",
                "TrafficSignFS",
                "CarFS",
                "PersonFS",
                "RiderFS",
                "OtherFS",
                "LeftLane",
                "TLGreenLeft",
                "FollowLeft",
                "NoLeftLane",
                "LeftObs",
                "LeftLine",
                "RightLane",
                "TLGreenRight",
                "FollowRight",
                "NoRightLane",
                "RightObs",
                "RightLine",
            ]
            assert len(names) == 21
            for i, name in enumerate(names):
                conf_matrix_w(i, name)

            stats_dict = {
                self.prefix + "/" + key: value
                for key, value in stats_dict.items()
            }
        stats_dict.update(super().create_dict(iterations))

        return stats_dict

class TrainLogger(Generic[LOG]):
    log: LOG

    def __init__(
        self,
        log_iterations: int,
        clazz: Type[LOG],
        args: Arguments,
        print_fn=None,
    ):
        self.clazz = clazz
        self.args = args
        self.print_fn = print_fn
        self.reset()
        if args.DEBUG:
            self.log_iterations = 1
        else:
            self.log_iterations = log_iterations
        self.iteration = 0
        self.epoch_stats = {}  # Track stats across the epoch

    def reset(self):
        self.log = self.clazz(self.args)

    def step(self):
        self.iteration += 1
        global GLOBAL_ITERATIONS
        GLOBAL_ITERATIONS += 1
        if self.iteration % self.log_iterations == 0:
            stats = self.log.create_dict(self.log_iterations)
            # Store stats for epoch-end printing
            self.epoch_stats = stats.copy()
            try:
                wandb.log(stats, step=GLOBAL_ITERATIONS)
            except Exception as e:
                print(f"Error logging stats: {e}")
            # Print stats if print function provided
            if self.print_fn is not None:
                self.print_fn(stats, "train")
            self.reset()

    def get_epoch_stats(self):
        """Return the most recent stats from this epoch."""
        return self.epoch_stats



class TestLogger(Generic[LOG]):
    log: LOG

    def __init__(self, clazz: Type[LOG], args, prefix: str, enable_wandb: bool=True):
        self.clazz = clazz
        self.prefix = prefix
        self.enable_wandb = enable_wandb
        self.args = args
        self.reset()

    def reset(self):
        self.log = self.clazz(self.args, self.prefix)

    def push(self, num_batches: int, extra_stats: dict={}):
        stats = self.log.create_dict(num_batches)
        if extra_stats is not None:
            extra_stats = {self.prefix + "/" + k: v for k, v in extra_stats.items()}
            stats.update(extra_stats)
        if self.enable_wandb:
            try:
                wandb.log(stats, step=GLOBAL_ITERATIONS)
            except Exception as e:
                print(f"Error logging stats: {e}")
        self.reset()
        return stats