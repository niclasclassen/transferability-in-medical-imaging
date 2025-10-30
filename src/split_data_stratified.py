# python split_data_stratified.py --dataset_name dermamnist --dataset_path ../data/original/dermamnist_224.npz --target_path ../splits/dermamnist_splits.npz
import argparse
import numpy as np
from medmnist import INFO
from typing import List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create data splits of a target dataset"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Choose one dataset from MedMNIST",
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path of the dataset location to sample from",
        required=True,
    )
    parser.add_argument(
        "--target_path",
        type=str,
        help="Path to where the samples will be saved.",
        required=True,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of how often the sampling will be repeated.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25, 50, 75, 100],
        help="List of percentages for data splits, e.g., 1 5 10 25",
    )

    return parser.parse_args()


def get_medmnist_data(dataset_name: str, dataset_path: str):
    metadata = INFO[dataset_name]
    data = np.load(dataset_path)

    return metadata, data


def stratified_sample(
    num_classes: int,
    labels: np.ndarray,
    split_percent: float,
    available_indices: Optional[np.ndarray] = None,
    prev_split: Optional[float] = None,
) -> np.ndarray:

    sampled_indices = []

    # If we're subsampling, compute how large a fraction of available_indices we need
    if available_indices is not None and prev_split is not None:
        fraction = split_percent / prev_split
    else:
        fraction = split_percent / 100.0  # sampling from full dataset

    for cls in range(num_classes):
        # Restrict to available indices if provided
        if available_indices is not None:
            cls_indices = [i for i in available_indices if labels[i] == cls]
        else:
            cls_indices = np.where(labels == cls)[0]

        n_samples = max(int(len(cls_indices) * fraction), 1)
        chosen = np.random.choice(cls_indices, n_samples, replace=False)
        sampled_indices.extend(chosen)

    return np.array(sampled_indices)


def create_samples(
    dataset_name: str,
    dataset_path: str,
    target_path: str,
    repetitions: int,
    splits: list,
):
    metadata, data = get_medmnist_data(dataset_name, dataset_path)
    num_classes = len(metadata["label"])

    savez_dict = {}

    # Save original data
    for key in data.files:
        savez_dict[key] = data[key]

    for rep in range(1, repetitions + 1):
        np.random.seed(42 + rep)

        prev_train_idx = None
        prev_val_idx = None
        prev_split = None

        for split in sorted(splits, reverse=True):

            # handle 100% edge case. Data already exists
            if split == 100:
                continue

            if prev_train_idx is None:
                # Largest split: sample from full dataset
                train_idx = stratified_sample(num_classes, data["train_labels"], split)
                val_idx = stratified_sample(num_classes, data["val_labels"], split)

            else:
                # Smaller split: stratified subsample from previous split
                train_idx = stratified_sample(
                    num_classes,
                    data["train_labels"],
                    split,
                    prev_train_idx,
                    prev_split,
                )
                val_idx = stratified_sample(
                    num_classes,
                    data["val_labels"],
                    split,
                    prev_val_idx,
                    prev_split,
                )

            prev_train_idx = train_idx
            prev_val_idx = val_idx
            prev_split = split

            # Save splits with keys identifying rep and split
            savez_dict[f"train_idx_run{rep}_split-{split}pct"] = train_idx
            savez_dict[f"val_idx_run{rep}_split-{split}pct"] = val_idx

    np.savez(target_path, **savez_dict)


def main():
    args = parse_args()
    create_samples(
        args.dataset_name,
        args.dataset_path,
        args.target_path,
        args.repetitions,
        args.splits,
    )


if __name__ == "__main__":
    main()
