# SPDX-License-Identifier: Apache-2.0
import os
import boto3
import argparse
import base64
import json
from botocore.exceptions import ClientError
import numpy as np
import gzip
import random
import matplotlib.pyplot as plt


def _convert_to_numpy(data_dir, images_file, labels_file):
    """Byte string to numpy arrays"""
    with gzip.open(os.path.join(data_dir, images_file), "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(data_dir, labels_file), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return (images, labels)


def mnist_to_numpy(data_dir="/tmp/data", train=True):
    """Download MNIST dataset and convert it to numpy array

    Args:
        data_dir (str): directory to save the data
        train (bool): download training set

    Returns:
        tuple of images and labels as numpy arrays
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if train:
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"

    # download objects
    s3 = boto3.client("s3")
    bucket = "sagemaker-sample-files"
    for obj in [images_file, labels_file]:
        key = os.path.join("datasets/image/MNIST", obj)
        dest = os.path.join(data_dir, obj)
        if not os.path.exists(dest):
            s3.download_file(bucket, key, dest)

    return _convert_to_numpy(data_dir, images_file, labels_file)


def download_npy_from_s3(data_dir="/tmp/data", train=True):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if train:
        images_file = "train_data.npy"
        labels_file = "train_labels.npy"
    else:
        images_file = "eval_data.npy"
        labels_file = "eval_labels.npy"

    # download objects
    s3 = boto3.client("s3")

    for obj in [images_file, labels_file]:
        key = os.path.join("tensorflow/mnist", obj)
        dest = os.path.join(data_dir, obj)
        if not os.path.exists(dest):
            s3.download_file("sagemaker-sample-data-ca-central-1", key, dest)
    return


def download_from_s3(data_dir="./data", train=True):
    """Download MNIST dataset and convert it to numpy array

    Args:
        data_dir (str): directory to save the data
        train (bool): download training set

    Returns:
        None
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if train:
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"

    # download objects
    s3 = boto3.client("s3")
    bucket = "sagemaker-sample-files"
    for obj in [images_file, labels_file]:
        key = os.path.join("datasets/image/MNIST", obj)
        dest = os.path.join(data_dir, obj)
        if not os.path.exists(dest):
            s3.download_file(bucket, key, dest)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    # CP4D specific arguments
    parser.add_argument("--catalog-id", type=str)
    parser.add_argument("--model-entry-id", type=str)
    parser.add_argument("--ibm-key-name", type=str, default="IBM_API_KEY_MLOPS")
    parser.add_argument("--cp4d-env", type=str, default=os.getenv("ENV", "saas"), choices=["saas", "prem"], )
    parser.add_argument("--cp4d-username", type=str, default=None)
    parser.add_argument("--cp4d-url", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="ptlt-mnist-model")
    parser.add_argument("--grc-model-name", type=str, default=None)
    # Training Job specific arguments (Sagemaker,Azure,WML) default SageMaker envar or Azure expected values
    parser.add_argument("--hosts", type=list, default=json.loads(os.getenv("SM_HOSTS", '["algo-1"]')))
    parser.add_argument("--current-host", type=str, default=os.getenv("SM_CURRENT_HOST", "algo-1"))
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./outputs"))
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAINING"))
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TESTING"))
    parser.add_argument("--validation", type=str, default=os.getenv("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--num-gpus", type=int, default=os.getenv("SM_NUM_GPUS"))

    # Model specific hyperparameters
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)")
    parser.add_argument("--learning-rate", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--beta_1", type=float, default=0.9, metavar="BETA1", help="beta1 (default: 0.9)")
    parser.add_argument("--beta_2", type=float, default=0.999, metavar="BETA2", help="beta2 (default: 0.999)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, metavar="WD", help="L2 weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--hidden-size", type=int, default=64, metavar="HIDDEN", help="size of hidden layer")
    parser.add_argument("--log-interval", type=int, default=100, metavar="N", help="how many batches to wait before logging training status")
    parser.add_argument("--backend", type=str, default=None, help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)")
    parser.add_argument("--model-type", type=str, default="fc", choices=["fc", "cnn"], metavar="MODEL", help="model type to be trained")

    # fmt: on
    return parser.parse_args()


def get_secret(
    secret_name: str,
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    region_name: str = "ca-central-1",
):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "AccessDeniedException":
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        else:
            print(e.response)
    else:
        if "SecretString" in get_secret_value_response:
            return get_secret_value_response["SecretString"]
        decoded_binary_secret = base64.b64decode(
            get_secret_value_response["SecretBinary"]
        )
        return decoded_binary_secret


def generate_data(
    data_path: str,
    num_samples: int = None,
    data_type: str = "test",
    img_cols: int = 28,
    img_rows: int = 28,
    channel_first: bool = False,
    show: bool = True,
):
    """
    to load the MNIST dataset from a specified path.

    :param data_path:str: Specify the path to the mnist data
    :param num_samples:int=None: Specify how many images should be sampled from the dataset
    :param data_type: Specify if the data being loaded is test or train
    :param img_cols:int=28: Set the width of the images
    :param img_rows:int=28: Specify the number of rows in the image
    :param channel_first:bool=True: Specify how image show be encoded
    :param show:bool=True: Show the first n images in a plot
    :return: x_ and y_ variables
    """

    if data_type in {"test", "validation"}:
        image_data = os.path.join(data_path, "t10k-images-idx3-ubyte.gz")
        label_data = os.path.join(data_path, "t10k-labels-idx1-ubyte.gz")

    elif data_type == "train":
        image_data = os.path.join(data_path, "train-images-idx3-ubyte.gz")
        label_data = os.path.join(data_path, "train-labels-idx1-ubyte.gz")
    else:
        raise ValueError("data_type should be test ,train or validation")

    x_, y_ = _convert_to_numpy(data_path, image_data, label_data)

    if channel_first:  # (1, img_rows, img_cols)
        x_ = x_.reshape(x_.shape[0], 1, img_rows, img_cols)
    else:  # (img_rows, img_cols, 1)
        x_ = x_.reshape(x_.shape[0], img_rows, img_cols, 1)

    x_ = x_.astype("float32")
    x_ /= 255

    if num_samples is None:
        return x_, y_
    # randomly sample n images to inspect
    mask = random.sample(range(x_.shape[0]), num_samples)
    samples = x_[mask]
    labels = y_[mask]

    if show:
        _plot_samples(samples)
    return samples, labels


def _plot_samples(samples):
    # plot the images
    fig, axs = plt.subplots(nrows=1, ncols=samples.shape[0], figsize=(16, 1))
    print(samples.shape, samples.dtype)
    for i, ax in enumerate(axs):
        ax.imshow(samples[i])
    plt.show()

    for i, ax in enumerate(axs):
        ax.imshow(samples[i])
    plt.show()
