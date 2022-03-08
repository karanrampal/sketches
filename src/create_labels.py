#!/usr/bin/env python3
"""Create labels"""

import argparse
import logging
import os

import pandas as pd
from utils.utils import set_logger


def args_parser() -> argparse.Namespace:
    """Parser for command line arguments"""
    parser = argparse.ArgumentParser(description="CLI")
    parser.add_argument("-r", "--root", type=str, default="../datasets/sketches", help="Root data dir")
    parser.add_argument("-i", "--img_path", type=str, default="images", help="Image dir")
    parser.add_argument("-d", "--data_path", type=str, default=".Training_data_Jan2022.xlsx", help="Data file")
    parser.add_argument("-e", "--file_ext", type=str, default=".png", help="File extension")
    parser.add_argument("-s", "--fill_str", type=str, default="None", help="String to fill null values")
    parser.add_argument("-l", "--log_path", type=str, default="../labels.log", help="Path of log file")
    parser.add_argument("-m", "--mode", type=str, default="mcml", help="Operation mode can be mc/mcml")
    parser.add_argument("-o", "--out_file", type=str, default="sketches_mlmc.csv", help="Output labels file")

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

    out.to_csv(os.path.join(args.root, args.out_file), index=False)


def multi_class(args: argparse.Namespace, data_filt: pd.DataFrame) -> None:
    """Writes multi-class labels to csv file
    Args:
        args: Hyperparams
        data_filt: Filtered dataframe
    """
    labels = pd.DataFrame(columns=["Image Id", "labels"])
    labels["Image Id"] = data_filt["Image Id"]
    labels["labels"] = data_filt[["Type", "Category", "SubCategory", "Customer Group"]].agg("-".join, axis=1)
    labels["code"] = labels["labels"].astype("category").cat.codes

    labels.to_csv(os.path.join(args.root, args.out_file), index=False)


def main() -> None:
    """Main function"""
    args = args_parser()
    set_logger(args.log_path)

    sketch_df = pd.read_excel(os.path.join(args.root, args.data_path))

    fetch_data = sketch_df.groupby("Image URL").first().reset_index()
    fetch_data = fetch_data[~(fetch_data["Image URL"] == "Image not found")]

    mask = fetch_data["Image Id"].duplicated()
    fetch_data["Image Id"] = fetch_data["Image Id"].astype("str")
    fetch_data.loc[mask, "Image Id"] += "_1"

    del_cols = ["Image URL", "Image Name", "Product Number", "Garment group", "Department Name", "Seasonold", "UniquieVal"]
    data = fetch_data.drop(del_cols, axis=1)

    data_clean = data.fillna(value=args.fill_str, axis=0)
    logging.info("Numberof unique types: %d", data_clean['Type'].nunique())
    logging.info("Numberof unique category: %d", data_clean['Category'].nunique())
    logging.info("Numberof unique subcategory: %d", data_clean['SubCategory'].nunique())
    logging.info("Numberof unique customer group: %d", data_clean['Customer Group'].nunique())

    file_list = os.listdir(os.path.join(args.root, args.img_path))
    id_list = [f[:-4] for f in file_list]
    data_filt = data_clean[data_clean["Image Id"].isin(id_list)]

    if args.mode == "mcml":
        multi_label(args, data_filt)
    elif args.mode == "mc":
        multi_class(args, data_filt)


if __name__ == "__main__":
    main()
