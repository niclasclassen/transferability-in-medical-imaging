# python split_data_stratified.py --dataset_name dermamnist --dataset_path ../data/original/dermamnist_224.npz --target_path ../splits/dermamnist_splits.npz
import argparse
import numpy as np
from medmnist import INFO
from typing import List


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
        default=3,
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
    num_classes: int, labels: np.ndarray, split_percent: int
) -> List[int]:
    sampled_indices = []

    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        n_samples = max(int(len(cls_indices) * (split_percent / 100)), 1)
        chosen = np.random.choice(cls_indices, n_samples, replace=False)
        sampled_indices.extend(chosen)

    return sampled_indices


def create_samples(
    dataset_name: str,
    dataset_path: str,
    target_path: str,
    repetitions: int,
    splits: list,
):
    metadata, data = get_medmnist_data(dataset_name, dataset_path)
    num_classes = len(metadata["label"])

    train_imgs = data["train_images"]
    train_labels = data["train_labels"]
    val_imgs = data["val_images"]
    val_labels = data["val_labels"]
    test_imgs = data["test_images"]  # keep as-is
    test_labels = data["test_labels"]  # keep as-is

    savez_dict = {}

    for rep in range(1, repetitions + 1):
        np.random.seed(42 + rep)

        for split in splits:
            # Sample train
            train_idx = stratified_sample(num_classes, train_labels, split)
            train_imgs_split = train_imgs[train_idx]
            train_labels_split = train_labels[train_idx]

            # Sample val
            val_idx = stratified_sample(num_classes, val_labels, split)
            val_imgs_split = val_imgs[val_idx]
            val_labels_split = val_labels[val_idx]

            # Save splits with keys identifying rep and split
            savez_dict[f"train_images_run{rep}_split-{split}pct"] = train_imgs_split
            savez_dict[f"train_labels_run{rep}_split-{split}pct"] = train_labels_split
            savez_dict[f"val_images_run{rep}_split-{split}pct"] = val_imgs_split
            savez_dict[f"val_labels_run{rep}_split-{split}pct"] = val_labels_split

    # Save test set as-is (no sampling)
    savez_dict["test_images"] = test_imgs
    savez_dict["test_labels"] = test_labels

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
