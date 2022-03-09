"""Utility functions for distributed computing"""

import errno
import logging
import os
from typing import Dict, Union
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
        with open(yaml_path, 'w', encoding='utf-8') as fptr:
            yaml.safe_dump(self.__dict__, fptr)

    def update(self, inp: Union[Dict, str]) -> None:
        """Loads parameters from yaml file or dict
        """
        if isinstance(inp, dict):
            self.__dict__.update(inp)
        elif isinstance(inp, str):
            with open(inp, encoding='utf-8') as fptr:
                params = yaml.safe_load(fptr)
                self.__dict__.update(params)
        else:
            raise TypeError(
            f"Input should either be a dictionary or a string path to a config file!"
        )

def set_logger(log_path):
    """Set the logger to log info in terminal and file at log_path.
    Args:
        log_path: (string) location of log file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model at checkpoint
    Args:
        state: (dict) contains model's state_dict, epoch, optimizer state_dict etc.
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    safe_makedir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model state_dict from checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def safe_makedir(path):
    """Make directory given the path if it doesn't already exist
    Args:
        path: path of the directory to be made
    """
    if not os.path.exists(path):
        print("Directory doesn't exist! Making directory {0}.".format(path))
        os.makedirs(path)
    else:
        print("Directory {0} Exists!".format(path))