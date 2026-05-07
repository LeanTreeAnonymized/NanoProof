import json
import os
import argparse
import shutil
import zipfile
from itertools import islice
import time
from pathlib import Path

import termplotlib as tpl
import numpy as np
import requests
from tqdm import tqdm

import leantree

from nanoproof.common import get_base_dir, format_distribution
from nanoproof.tokenizer import get_tokenizer

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "leantree")
JSONL_FILENAME = "leantree_mathlib.jsonl"

# Original HuggingFace URL withheld for double-blind review; an anonymized
# mirror of the dataset is hosted on Kaggle and used during the review period.
KAGGLE_DATASET_SLUG = "leantree/leantree"
KAGGLE_ARCHIVE_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET_SLUG}"
)
KAGGLE_PAGE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_SLUG}/data"


def leantree_transitions(split, eval_fraction=0.1, augmentations=None):
    assert split in ("train", "valid"), f"Invalid split: {split!r}"
    mathlib_file = os.path.join(DATA_DIR, "leantree_mathlib.jsonl")
    if not Path(mathlib_file).exists():
        raise Exception(
            "leantree not downloaded, please run this script with `download` argument"
        )
    with open(mathlib_file, "r") as f:
        lines = f.readlines()
    eval_size = int(len(lines) * eval_fraction)
    lines = lines[:-eval_size] if split == "train" else lines[-eval_size:]

    for line in lines:
        lean_file = leantree.LeanFile.deserialize(json.loads(line))
        for thm in lean_file.theorems:
            if isinstance(thm, leantree.StoredError):
                continue
            for by_block in thm.by_blocks:
                if isinstance(by_block.tree, leantree.StoredError):
                    continue
                for node in by_block.tree.get_nodes():
                    if augmentations:
                        for aug in augmentations:
                            node = aug.run(node)
                    tactic = str(node.tactic.tactic)
                    if "sorry" in tactic or "admit" in tactic:
                        # shouldn't happen, but just in case
                        continue
                    if tactic.strip() == "bound":
                        # `bound` tactic messes with the kernel check
                        continue
                    yield str(node.state), tactic, node.proof_depth


def _print_manual_instructions(reason: str) -> None:
    print(
        "\nCould not auto-download the LeanTree SFT dataset "
        f"({reason}).\n"
        "Please download it manually:\n"
        f"  1. Open {KAGGLE_PAGE_URL}\n"
        '  2. Click "Download" (a Kaggle account may be required).\n'
        f"  3. Unzip the archive and place {JSONL_FILENAME} at:\n"
        f"       {os.path.join(DATA_DIR, JSONL_FILENAME)}\n"
    )


def _download_from_kaggle(jsonl_path: str) -> bool:
    """Try to fetch leantree_mathlib.jsonl from the public Kaggle archive
    without authentication. Returns True on success, False otherwise.
    Prints manual-download instructions on failure.
    """
    archive_path = jsonl_path + ".zip.tmp"
    try:
        print(f"Attempting Kaggle download: {KAGGLE_ARCHIVE_URL}")
        with requests.get(
            KAGGLE_ARCHIVE_URL, stream=True, timeout=60, allow_redirects=True
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "zip" not in content_type and "octet-stream" not in content_type:
                raise RuntimeError(
                    f"unexpected content-type {content_type!r} "
                    "(Kaggle likely returned an HTML login page)"
                )
            total_size = int(response.headers.get("content-length", 0))
            with open(archive_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="kaggle archive",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
    except (requests.RequestException, RuntimeError) as e:
        if os.path.exists(archive_path):
            os.remove(archive_path)
        _print_manual_instructions(f"Kaggle request failed: {e}")
        return False

    try:
        with zipfile.ZipFile(archive_path) as zf:
            members = [m for m in zf.namelist() if m.endswith(JSONL_FILENAME)]
            if not members:
                raise RuntimeError(
                    f"{JSONL_FILENAME} not found inside Kaggle archive"
                )
            tmp_path = jsonl_path + ".tmp"
            with zf.open(members[0]) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.rename(tmp_path, jsonl_path)
        os.remove(archive_path)
        print(f"Successfully downloaded {jsonl_path}")
        return True
    except (zipfile.BadZipFile, RuntimeError, OSError) as e:
        for path in (archive_path, jsonl_path + ".tmp"):
            if os.path.exists(path):
                os.remove(path)
        _print_manual_instructions(f"failed to extract archive: {e}")
        return False


def download_dataset():
    """Download the leantree SFT dataset.

    During the double-blind review period the dataset is mirrored on Kaggle
    under an anonymous slug; we try to fetch it automatically and fall back
    to printing manual-download instructions if Kaggle requires auth or the
    archive layout has changed.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    jsonl_path = os.path.join(DATA_DIR, JSONL_FILENAME)
    if os.path.exists(jsonl_path):
        print(f"Already downloaded: {jsonl_path}")
        return
    if not _download_from_kaggle(jsonl_path):
        raise SystemExit(1)


def _print_stats():
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    assert bos_token is not None
    eos_token = tokenizer.get_eos_token_id()
    assert eos_token is not None
    for split in ("train", "valid"):
        print(f"Loading {split=}...")
        dataset = list(leantree_transitions(split=split))
        print(f"Calculating {split=}...")
        lens = {"state": [], "tactic": []}
        depths = []
        start_time = time.time()
        for state, tactic, proof_depth in tqdm(dataset):
            state = tokenizer.encode(state + "\n<|tactic|> ", prepend=bos_token)
            tactic = tokenizer.encode(tactic, append=eos_token)
            lens["state"].append(len(state))
            lens["tactic"].append(len(tactic))
            depths.append(proof_depth)
        end_time = time.time()
        print(f"time: {end_time - start_time:.2f}s")
        print(f"total: {len(lens['state'])}")
        for prop, max_len in [("state", 448), ("tactic", 64)]:
            print(f"{prop} lengths:")
            print(f"  min: {np.min(lens[prop])}")
            print(f"  max: {np.max(lens[prop])}")
            print(f"  mean: {np.mean(lens[prop]):.2f}")
            print(f"  median: {np.median(lens[prop])}")
            print(f"  std: {np.std(lens[prop]):.2f}")
            print(f"  p90: {np.percentile(lens[prop], 90):.2f}")
            print(f"  p95: {np.percentile(lens[prop], 95):.2f}")
            print(f"  p99: {np.percentile(lens[prop], 99):.2f}")
            at_most_max = np.sum(np.array(lens[prop]) <= max_len)
            print(
                f"  <= {max_len}: {at_most_max / len(lens[prop]):%} ({at_most_max}/{len(lens[prop])})"
            )
        print(f"depths:")
        print(f"  min: {np.min(depths)}")
        print(f"  max: {np.max(depths)}")
        print(f"  mean: {np.mean(depths):.2f}")
        print(f"  median: {np.median(depths)}")
        print(f"  p90: {np.percentile(depths, 90):.2f}")
        print(f"  p95: {np.percentile(depths, 95):.2f}")
        print(f"  p99: {np.percentile(depths, 99):.2f}")
        at_most_32 = np.sum(np.array(depths) <= 32)
        print(f"  <= 32: {at_most_32 / len(depths):%} ({at_most_32}/{len(depths)})")
        print()

        fig = tpl.figure()
        min_depth = int(np.min(depths))
        max_depth = int(np.max(depths))
        bin_edges = np.arange(
            min_depth, max_depth + 2
        )  # +2 to include max_depth in a bin
        counts, bin_edges = np.histogram(depths, bins=bin_edges)
        fig.hist(
            counts, bin_edges=bin_edges, force_ascii=False, orientation="horizontal"
        )
        fig.show()
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download LeanTree dataset from HuggingFace.", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(dest="action")

    download_parser = subparsers.add_parser("download")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--split", choices=["train", "valid"], default="train")

    stats_parser = subparsers.add_parser("stats")

    args = parser.parse_args()

    if args.action == "download":
        os.makedirs(DATA_DIR, exist_ok=True)
        download_dataset()
    elif args.action == "show":
        for state, tactic, _ in islice(leantree_transitions(split=args.split), 10):
            print(state)
            print("\n->\n")
            print(tactic)
            print("\n-----------------\n")
    elif args.action == "stats":
        _print_stats()
    else:
        raise ValueError(f"Unknown action {args.action}")


if __name__ == "__main__":
    main()
