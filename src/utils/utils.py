"""Utility functions for distributed computing"""

from collections import deque
import errno
import logging
import os
from typing import Any, Dict, Optional, Union
import shutil

import torch
import yaml


class Params:
    """Class to load hyperparameters from a yaml file.
    """
    def __init__(self, inp: Union[Dict, str]) -> None:
        self.update(inp)

    def save(self, yaml_path: str) -> None:
        """Save parameters to yaml file at yaml_path
        """
        with open(yaml_path, "w", encoding="utf-8") as fptr:
            yaml.safe_dump(self.__dict__, fptr)

    def update(self, inp: Union[Dict, str]) -> None:
        """Loads parameters from yaml file or dict
        """
        if isinstance(inp, dict):
            self.__dict__.update(inp)
        elif isinstance(inp, str):
            with open(inp, encoding="utf-8") as fptr:
                params = yaml.safe_load(fptr)
                self.__dict__.update(params)
        else:
            raise TypeError(
            f"Input should either be a dictionary or a string path to a config file!"
        )

def set_logger(log_path: str) -> None:
    """Set the logger to log info in terminal and file at log_path.
    Args:
        log_path: Location of log file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint: str) -> None:
    """Saves model at checkpoint
    Args:
        state: Contains model's state_dict, epoch, optimizer state_dict etc.
        is_best: True if it is the best model seen till now
        checkpoint: Folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, "last.pth.tar")
    safe_makedir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))

def load_checkpoint(checkpoint: str, model: torch.nn.Module, optimizer: torch.optim = None):
    """Loads model state_dict from checkpoint.
    Args:
        checkpoint: Filename which needs to be loaded
        model: Model for which the parameters are loaded
        optimizer: Resume optimizer from checkpoint, optional
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint

def safe_makedir(path: str) -> None:
    """Make directory given the path if it doesn't already exist
    Args:
        path: Path of the directory to be made
    """
    if not os.path.exists(path):
        print(f"Directory doesn't exist! Making directory {path}.")
        os.makedirs(path)
    else:
        print(f"Directory {path} Exists!")


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None) -> None:
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: Any, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> Union[int, float]:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> Union[int, float]:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> Union[int, float]:
        return self.total / self.count

    @property
    def max(self) -> Union[int, float]:
        return max(self.deque)

    @property
    def value(self) -> Union[int, float]:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )

def save_dict_to_yaml(data, yml_path):
    """Saves a dict of floats to yaml file
    Args:
        data: (dict) of float-castable values (np.float, int, float, etc.)
        yml_path: (string) path to yaml file
    """
    with open(yml_path, 'w', encoding='utf-8') as fptr:
        data = {k: float(v) for k, v in data.items()}
        yaml.safe_dump(data, fptr)
