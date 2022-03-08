"""Download data utility functions"""

import io
import logging
import os
import requests

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def fetch_images(
    data: pd.DataFrame,
    dir_path: str,
    img_path: str,
    img_ext: str = "jpg"
) -> None:
    """Fetch images given the dataframe of urls and image id's
    Args:
        data: Input dataframe containing image urls and id's
        dir_path: Root directory
        img_path: Path to save images to
        img_ext: Image extension to save as
    """
    tot_url = len(data)

    cnt_downloaded, cnt_exists, cnt_resp_err = 0, 0, 0
    for _, row in tqdm(data.iterrows(), unit="rows", total=tot_url):
        url = row["Image URL"]
        id_ = row["Image Id"]
        file_name = os.path.join(dir_path, img_path, f"{id_}.{img_ext}")
        if not os.path.isfile(file_name):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    with Image.open(io.BytesIO(response.content)) as img:
                        try:
                            img.convert("RGB").save(file_name)
                        except OSError:
                            img.convert("RGB").save(file_name)
                    cnt_downloaded += 1
                else:
                    cnt_resp_err += 1
            except requests.exceptions.RequestException as e:
                raise SystemExit(e)
        else:
            cnt_exists += 1

    logging.info("Images downloaded: %d!", cnt_downloaded)
    logging.info("Non existent urls: %d!", cnt_resp_err)
    logging.info("Images already exist: %d!", cnt_exists)
