# SPDX-License-Identifier: Apache-2.0
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from use_case_mnist_pt.pytorch_inference import (
    model_fn,
    input_fn,
    predict_fn,
    output_fn,
)
from utils import mnist_to_numpy
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    "--inference-samples",
    type=int,
    default=2,
    metavar="NSAMPLES",
    help="Number of samples to be sent for inference",
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running Inference for Pytorch Model {args}")
    data_dir = "/tmp/data"

    X, Y = mnist_to_numpy(data_dir, train=False)

    # randomly sample 16 images to inspect
    mask = random.sample(range(X.shape[0]), args.inference_samples)
    samples = X[mask]
    labels = Y[mask]
    # plot the images
    fig, axs = plt.subplots(nrows=1, ncols=args.inference_samples, figsize=(16, 1))

    for i, ax in enumerate(axs):
        ax.imshow(samples[i])
    plt.show()
    print(samples.shape)
    samples = np.expand_dims(samples, axis=1)

    print(samples.shape)

    inputs = {"input_data": [{"values": samples.tolist()}]}

    model = model_fn("../models")
    print(samples.shape, samples.dtype)
    print(json.dumps(inputs))
    input_tensors = input_fn(json.dumps(inputs), "application/json")
    print(input_tensors.shape)
    outputs = predict_fn(input_tensors, model)
    print(outputs.shape)
    preds = output_fn(outputs, "application/json")
    print(preds)
