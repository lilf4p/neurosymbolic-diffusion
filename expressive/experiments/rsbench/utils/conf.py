# config module

import random
import torch
import numpy as np
import os


def get_device(args=None):
    if args is None or not hasattr(args, "use_cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check for available GPUs
    use_cuda = args.use_cuda and torch.cuda.is_available()
    # Check for MPS (Metal Performance Shaders) availability
    use_mps = args.use_mps and torch.backends.mps.is_available()
    if use_mps:
        # set fallback to True
        import os

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
    elif use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    if args.DEBUG:
        torch.autograd.set_detect_anomaly(True)
        print("Debugging mode")
    return device


def base_path() -> str:
    """Returns the base bath where to log accuracies and tensorboard data.

    Returns:
        base_path (str): base path
    """
    return "./data/"


def set_random_seed(seed: int) -> None:
    """Sets the seeds at a certain value.

    Args:
        param seed: the value to be set

    Returns:
        None: This function does not return a value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_path(path) -> None:
    """Create path function, create folder if it does not exists

    Args:
        path (str): path value

    Returns:
        None: This function does not return a value.
    """
    if not os.path.exists(path):
        os.makedirs(path)
