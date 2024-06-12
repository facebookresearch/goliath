# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset download
"""

import argparse
import json
import logging
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Union
from urllib.error import HTTPError, URLError

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

BPATH = "https://s3.us-west-2.amazonaws.com/codec-avatars-oss/goliath-4/4TB/"


def load_links(links_path: Union[str, Path]) -> List[str]:
    with open(links_path, "r") as f:
        links = f.readlines()  
    return links


# def download_link(from_url: str, to_path: str) -> None:
def download_link(from_url_and_to_path: Tuple[int, int, str, str]) -> None:
    """Download a single link"""
    i, total, from_url, to_path = from_url_and_to_path

    to_path.parent.mkdir(parents=True, exist_ok=True)

    if to_path.exists():
        # TODO(julieta) Local file could be corrupted or outdated, check hashes instead of just skipping.
        logging.info("%s already exists, skipping", to_path)
        return

    percent_done = 100 * i / total
    logging.info("[%.2f%%] Downloading link %d / %d from %s to %s", percent_done, i, total, from_url, to_path)

    try:
        urllib.request.urlretrieve(from_url, to_path)
    except HTTPError as e:
        logging.error("HTTP error occurred reaching %s: %s", from_url, e)
        raise e
    except URLError as e:
        logging.error("URL error occurred reaching %s: %s", from_url, e)
        raise e


def download_links(links_and_paths: List[Tuple[int, int, str, str]]) -> None:
    """Download a bunch of links to a series of filesystem paths"""
    for i, total, from_url, to_path in links_and_paths:
        download_link((i, total, from_url, to_path))


def main():
    parser = argparse.ArgumentParser(description="Download the goliath dataset")
    parser.add_argument("--links-file", type=str, default="signed_10.txt", help="CSV file with captures to download")
    parser.add_argument("--output-dir", "-o", type=str, help=f"Directory to write the dataset to", required=True)
    parser.add_argument("-n", type=int, default=7645, help="Number of links to download from links-file")
    parser.add_argument("--workers", "-j", type=int, default=8, help="Number of workers for parallel download")
    # TODO(julieta) let people pass a single sid
    # TODO(julieta) check the hash of the remote files and compare with local files
    args = parser.parse_args()

    # TODO(julieta) check version match, if mismatch, then delete/download new data

    # Check links file
    links = load_links(args.links_file)
    if args.n > len(links):
        raise ValueError(f"Requested more links ({args.n}) than available in captures file ({len(links)})")
    logging.info(
        "Downloading the first %d out of %d links from %s",
        args.n,
        len(links),
        args.links_file,
    )
    links = links[: args.n]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    links_and_paths: List[Tuple] = []

    # TODO(julieta) check that these are valid captures, sid mcd and mct are there
    for link in links:
        from_url = link

        # NOTE(julieta) Parse url to infer destination path
        unsigned_url = from_url.split("?X-Amz-Algorithm")[0]
        raw_path = unsigned_url[len(BPATH):]
        to_path = output_dir / raw_path
        print(to_path)
        
        links_and_paths.append((from_url, to_path))


    # Done creating links, donwload everything
    total_links = len(links_and_paths)
    links_and_paths = [(i + 1, total_links, link, path) for i, (link, path) in enumerate(links_and_paths)]

    n_workers = min(args.workers, cpu_count())
    logging.info("Downloading %s files with %s workers", len(links_and_paths), n_workers)

    if n_workers == 1:
        logging.warning("Downloading with a single worker. This might be slow, consider using more workers.")
        download_links(links_and_paths)
    else:
        pool = Pool(n_workers)
        download_func = partial(download_link)
        pool.imap(download_func, links_and_paths)
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
