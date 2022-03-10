#!/usr/bin/env python3
"""Download data"""

import argparse
import logging
import os

import pandas as pd

from utils.download_utils import fetch_images
from utils.utils import set_logger


def args_parser() -> argparse.Namespace:
    """Parser for command line arguments"""
    parser = argparse.ArgumentParser(description="CLI")
    parser.add_argument("-r", "--root", type=str, default="../datasets/sketches", help="Root data dir")
    parser.add_argument("-i", "--img-path", type=str, default="images", help="Image dir")
    parser.add_argument("-d", "--data-path", type=str, default="Training_data_Jan2022.xlsx", help="Data file")
    parser.add_argument("-e", "--file-ext", type=str, default=".png", help="File extension")
    parser.add_argument("-l", "--log-path", type=str, default="download.log", help="Path of log file")

    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = args_parser()
    set_logger(args.log_path)

    logging.info("Reading XLSX file...")
    sketch_df = pd.read_excel(os.path.join(args.root, args.data_path))
    logging.info("Unique URL's %d", sketch_df["Image URL"].nunique())
    logging.info("Unique ID's %d", sketch_df["Image Id"].nunique())
    logging.info("Unique Product number's %d", sketch_df["Product Number"].nunique())
    logging.info("Unique Name's %d", sketch_df["Image Name"].nunique())

    logging.info("Removing invalid URL's...")
    fetch_data = sketch_df.groupby("Image URL").first().reset_index()
    logging.info(f"Unique URL's: %d", fetch_data.shape[0])
    fetch_data = fetch_data[~(fetch_data["Image URL"] == "Image not found")]
    logging.info(f"Valid unique URL's: %d", fetch_data.shape[0])

    logging.info(f"Unique URL's %d", fetch_data["Image URL"].nunique())
    logging.info(f"Unique ID's %d", fetch_data["Image Id"].nunique())

    logging.info("Fix image ID's...")
    mask = fetch_data["Image Id"].duplicated()
    fetch_data["Image Id"] = fetch_data["Image Id"].astype("str")
    fetch_data.loc[mask, "Image Id"] += "_1"
    logging.info(f"Unique URL's %d", fetch_data["Image URL"].nunique())
    logging.info(f"Unique ID's %d", fetch_data["Image Id"].nunique())

    logging.info("Downloading images...")
    os.makedirs(os.path.join(args.root, args.img_path), exist_ok=True)
    fetch_images(fetch_data, args.root, args.img_path, args.file_ext)


if __name__ == "__main__":
    main()
