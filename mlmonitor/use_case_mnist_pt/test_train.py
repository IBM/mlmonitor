# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from utils import parse_args
from use_case_mnist_pt.pytorch_train import train
from dotenv import load_dotenv
from use_case_mnist_tf.utils import download_from_s3

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


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

    load_dotenv()
    Env()

    download_from_s3(f"{PROJECT_ROOT}/datasets/mnist", True)
    download_from_s3(f"{PROJECT_ROOT}/datasets/mnist", False)

    args = parse_args()
    # Clean up MLFLOW and lighting_logs
    for dir in ["./mlruns", "./lighting_logs", "./logs"]:
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    args = vars(args)
    print(args)
    train(args)
