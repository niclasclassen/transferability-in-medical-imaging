#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_flag", type=str, help="choose one dataset from MedMNIST datasets"
)

parser.add_argument(
    "--source_path",
    type=str,
    help="add path of the source MedMNIST dataset you want to sample from",
)

parser.add_argument(
    "--target_path",
    type=str,
    help="add path of the target location you want to save the sample",
)
args = parser.parse_args()

import numpy as np
from medmnist import INFO


def make_splits(data_flag, source_path, target_path):
    savez_dict = dict()
    info = INFO[data_flag]
    num_class = len(info["label"])
    test_data = np.load(source_path)
    train_imgs = test_data["train_images"]
    train_labels = test_data["train_labels"]
    val_imgs = test_data["val_images"]
    val_labels = test_data["val_labels"]
    np.random.seed(24)
    for i in range(1, 6):
        idx = np.random.choice(train_imgs.shape[0], num_class * 100, replace=False)
        train_imgs_fold = train_imgs[idx, :, :]
        train_labels_fold = train_labels[idx, :]
        savez_dict["train_imgs_fold" + str(i)] = train_imgs_fold
        savez_dict["train_labels_fold" + str(i)] = train_labels_fold
        if len(val_labels) > num_class * 25:
            idx = np.random.choice(val_imgs.shape[0], num_class * 25, replace=False)
            val_imgs_fold = val_imgs[idx, :, :]
            val_labels_fold = val_labels[idx, :]
        else:
            val_imgs_fold = val_imgs
            val_labels_fold = val_labels
        savez_dict["val_imgs_fold" + str(i)] = val_imgs_fold
        savez_dict["val_labels_fold" + str(i)] = val_labels_fold
    np.savez(target_path + data_flag + "_fine-tune_224.npz", **savez_dict)


make_splits(args.data_flag, args.source_path, args.target_path)
