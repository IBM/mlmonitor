# SPDX-License-Identifier: Apache-2.0
import numpy as np
import gzip
import os
import torch
from torch.utils.data import Dataset


def normalize(x, axis):
    eps = np.finfo(float).eps
    mean = np.mean(x, axis=axis, keepdims=True)
    # avoid division by zero
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std


def convert_to_tensor(data_dir, images_file, labels_file):
    """Byte string to torch tensor"""
    with gzip.open(os.path.join(data_dir, images_file), "rb") as f:
        images = (
            np.frombuffer(f.read(), np.uint8, offset=16)
            .reshape(-1, 28, 28)
            .astype(np.float32)
        )

    with gzip.open(os.path.join(data_dir, labels_file), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    # normalize the images
    images = normalize(images, axis=(1, 2))

    # add channel dimension (depth-major)
    images = np.expand_dims(images, axis=1)

    # to torch tensor
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    return images, labels


class MNIST(Dataset):
    def __init__(self, data_dir, train=True):

        if train:
            images_file = "train-images-idx3-ubyte.gz"
            labels_file = "train-labels-idx1-ubyte.gz"
        else:
            images_file = "t10k-images-idx3-ubyte.gz"
            labels_file = "t10k-labels-idx1-ubyte.gz"

        self.images, self.labels = convert_to_tensor(data_dir, images_file, labels_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
