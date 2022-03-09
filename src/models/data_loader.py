"""Define custom dataset class extending the Pytorch Dataset class"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tvt

from utils.utils import Params


class SketchesDataset(Dataset):
    """Custom class for Sketches dataset"""

    def __init__(self, root: str, csv_file: str, transform: tvt = None) -> None:
        """Get the filenames and labels of images from a csv file.
        Args:
            root: Directory containing the data
            csv_file: file containing the data
            transform: Transformation to apply on images
        """
        self.root = root
        self.data = pd.read_csv(os.path.join(root, csv_file))
        self.transform = transform

    def __len__(self) -> int:
        """Return the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, np.ndarray]:
        """Get an item from the dataset given the index idx"""
        row = self.data.iloc[idx]

        im_name = row["Image Id"] + ".png"
        im_path = os.path.join(self.root, im_name)
        img = Image.open(im_path).convert("RGB")

        labels = row[1:].values

        if self.transform is not None:
            img = self.transform(img)

        return img, labels


def get_transform(mode: str, params: Params) -> tvt.Compose:
    """Data augmentation
    Args:
        is_train: If the dataset is training
    Returns:
        Composition of all the data transforms
    """
    trans = []
    trans.append(
        tvt.Resize((params.height, params. width)),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    if mode == "train":
        trans.append(
            tvt.RandomHorizontalFlip(params.flip),
            tvt.ColorJitter(
                brightness=params.brightness,
                contrast=params.contrast,
                saturation=params.saturation,
                hue=params.hue
            ),
            tvt.RandomRotation(params.degree)
        )
    return tvt.Compose(trans)


def get_dataloader(
    modes: List[str],
    params: Params,
) -> Dict[str, DataLoader]:
    """Get DataLoader objects.
    Args:
        modes: Mode of operation i.e. 'train', 'val', 'test'
        params: Hyperparameters
    Returns:
        DataLoader object for each mode
    """
    dataloaders = {}

    for mode in modes:
        if mode == "train":
            trans = get_transform(mode, params)
            shuf = True
        else:
            trans = get_transform(mode, params)
            shuf = False

        dataset = SketchesDataset(
            root=params.data_path,
            csv_file=mode + "_sketches_" + params.type + ".csv",
            transform=trans
        )
        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            shuffle=shuf
        )

    return dataloaders
