# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from os.path import dirname
from dotenv import load_dotenv

# from tf_train import train_model as train_job  # Fully connected model
from tf_cnn_train import train_job

# from tf_train import _parse_args  # Fully connected model
from tf_cnn_train import _parse_args

from use_case_mnist_tf.utils import download_from_s3

PROJECT_ROOT = dirname(dirname(__file__))


class Env:
    def __init__(self):
        # simulate container env
        os.environ["SM_MODEL_DIR"] = os.path.join(PROJECT_ROOT, "models", "model_mnist")
        os.environ["SM_CHANNEL_TRAINING"] = f"{PROJECT_ROOT}/datasets/mnist"
        os.environ["SM_CHANNEL_TESTING"] = f"{PROJECT_ROOT}/datasets/mnist"
        os.environ["SM_HOSTS"] = '["algo-1"]'
        os.environ["SM_CURRENT_HOST"] = "algo-1"
        os.environ["SM_NUM_GPUS"] = "0"


if __name__ == "__main__":

    download_from_s3(f"{PROJECT_ROOT}/datasets/mnist", True)
    download_from_s3(f"{PROJECT_ROOT}/datasets/mnist", False)

    load_dotenv()
    Env()

    # Clean up MLFLOW and lighting_logs
    for dir in ["./mlruns", "./lighting_logs", "./logs"]:
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    args, unknown = _parse_args()

    args = vars(args)
    print(args)
    train_job(args)
