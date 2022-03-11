#!/usr/bin/env python3
"""Evaluates the model"""

import argparse
import logging
import os
from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils.utils as utils
from model.net import Net, loss_fn, get_metrics
from model.data_loader import get_dataloader


def args_parser() -> argparse.Namespace:
    """Parse commadn line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="../datasets/",
                        help="Directory containing the dataset")
    parser.add_argument("--model_dir",
                        default="experiments/base_model",
                        help="Directory containing params.json")
    parser.add_argument("--restore_file",
                        default="last",
                        choices=["last", "best"],
                        help="name of the file in --model_dir containing weights to load")
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    criterion: Callable,
    dataloader: DataLoader,
    metrics: Dict[str, Any],
    params: utils.Params,
    writer: SummaryWriter,
    epoch: int
) -> Dict[str, Any]:
    """Evaluate the model on `num_steps` batches.
    Args:
        model: Neural network
        criterion: A function that computes the loss for the batch
        dataloader: Test dataloader
        metrics: A dictionary of functions that compute a metric
        params: Hyperparameters
        writer : Summary writer for tensorboard
        epoch: Value of Epoch
    """
    # put model in evaluation mode
    model.eval()
    summ = []

    with torch.no_grad():
        for i, (inp_data, labels) in enumerate(dataloader):
            # move data to GPU if possible
            if params.cuda:
                inp_data = inp_data.to(params.device)
                labels = labels.to(params.device)

            # compute model output
            output = model(inp_data)
            loss = criterion(output, labels, params)

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output, labels) for metric in metrics}
            summary_batch["loss"] = loss.item()
            summ.append(summary_batch)

            # Add to tensorboard
            writer.add_scalar("testing_loss", summary_batch["loss"],
                              epoch * len(dataloader) + i)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join(f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Eval metrics : %s", metrics_string)
    return metrics_mean


def main() -> None:
    """Main function
    """
    args = args_parser()
    params_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(params_path), f"No json configuration file found at {params_path}"
    params = utils.Params(params_path)
    params.update(vars(args))

    writer = SummaryWriter(os.path.join(args.model_dir, "runs", "eval"))

    params.cuda = torch.cuda.is_available()

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        params.device = "cuda:0"
    else:
        params.device = "cpu"

    utils.set_logger(os.path.join(args.model_dir, "evaluate.log"))

    logging.info("Loading the dataset...")

    dataloaders = get_dataloader(["test"], params)
    test_dl = dataloaders["test"]

    logging.info("- done.")

    model = Net(params)
    if params.cuda:
        model = model.to(params.device)
    writer.add_graph(model, next(iter(test_dl))[0])

    criterion = loss_fn
    metrics = get_metrics()

    logging.info("Starting evaluation")

    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + ".pth.tar"), model)

    evaluate(model, criterion, test_dl, metrics, params, writer, 0)

    writer.close()


if __name__ == "__main__":
    main()
