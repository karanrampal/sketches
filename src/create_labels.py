#!/usr/bin/env python3
"""Create labels"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.utils import set_logger


def args_parser() -> argparse.Namespace:
    """Parser for command line arguments"""
    parser = argparse.ArgumentParser(description="CLI")
    parser.add_argument("-r", "--root", type=str, default="../datasets/sketches", help="Root data dir")
    parser.add_argument("-i", "--img-path", type=str, default="images", help="Image dir")
    parser.add_argument("-d", "--data-path", type=str, default="Training_data_Jan2022.xlsx", help="Data file")
    parser.add_argument("-s", "--fill-str", type=str, default="None", help="String to fill null values")
    parser.add_argument("-l", "--log-path", type=str, default="labels.log", help="Path of log file")
    parser.add_argument("-m", "--mode", type=str, default="mcml", help="Operation mode can be mc/mcml")

    return parser.parse_args()


def multi_label(args: argparse.Namespace, data_filt: pd.DataFrame) -> None:
    """Writes multi-class multi-label labels to csv file
    Args:
        args: Hyperparams
        data_filt: Filtered dataframe
    """
    out = pd.get_dummies(
        data_filt,
        columns=["Type", "Category", "SubCategory", "Customer Group"],
    )

    tmp = out.groupby("Image Id").sum()
    assert (tmp > 1).any().any() == False
    assert (tmp < 0).any().any() == False
    assert (tmp == 0).all().any() == False
    assert (tmp == 0).all(1).any() == False
    assert (tmp.sum(1) > 1).all() == True
    assert tmp.reset_index().shape == out.shape

    logging.info("Writing CSV file...")
    out_train, out_val = train_test_split(out, test_size=0.1, random_state=42)
    fname = "_sketches_" + args.mode + ".csv"
    out_train.to_csv(os.path.join(args.root, "train" + fname), index=False)
    out_val.to_csv(os.path.join(args.root, "val" + fname), index=False)


def main() -> None:
    """Main function"""
    np.random.seed(31415)

    args = args_parser()
    set_logger(args.log_path)

    logging.info("Reading XLSX file...")
    sketch_df = pd.read_excel(os.path.join(args.root, args.data_path))

    logging.info("Removing 'Image not found' URL's...")
    fetch_data = sketch_df.groupby("Image URL").first().reset_index()
    fetch_data = fetch_data[~(fetch_data["Image URL"] == "Image not found")]

    logging.info("Fixing Image Id's...")
    mask = fetch_data["Image Id"].duplicated()
    fetch_data["Image Id"] = "img_" + fetch_data["Image Id"].astype("str")
    fetch_data.loc[mask, "Image Id"] += "_1"

    del_cols = ["Image URL", "Image Name", "Product Number", "Garment group", "Department Name", "Seasonold", "UniquieVal"]
    data = fetch_data.drop(del_cols, axis=1)

    logging.info("Filling NAN's...")
    data_clean = data.fillna(value=args.fill_str, axis=0)
    logging.info("Numberof unique types: %d", data_clean['Type'].nunique())
    logging.info("Numberof unique category: %d", data_clean['Category'].nunique())
    logging.info("Numberof unique subcategory: %d", data_clean['SubCategory'].nunique())
    logging.info("Numberof unique customer group: %d", data_clean['Customer Group'].nunique())

    logging.info("Filter data...")
    file_list = os.listdir(os.path.join(args.root, args.img_path))
    id_list = [f[:-4] for f in file_list]
    data_filt = data_clean[data_clean["Image Id"].isin(id_list)]

    multi_label(args, data_filt)


if __name__ == "__main__":
    main()
