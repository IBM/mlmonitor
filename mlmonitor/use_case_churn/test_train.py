# SPDX-License-Identifier: Apache-2.0
import boto3
import sagemaker
from use_case_churn.train_cc_xg_boost import (
    logger,
    _parse_args,
    run_train_job,
)
from dotenv import load_dotenv
import tempfile
import tarfile
import gzip
import os
import shutil
from os.path import dirname

PROJECT_ROOT = dirname(dirname(__file__))


def make_model_tgzfile(output_filename: str, source_dir: str):
    temp_unzipped_dir = tempfile.mktemp()
    current_dir = os.path.abspath(os.getcwd())
    assert os.path.exists(source_dir)
    os.chdir(source_dir)
    try:
        with tarfile.open(temp_unzipped_dir, "w") as tar:
            for filename in os.listdir(source_dir):
                if filename.startswith("xgboost"):
                    tar.add(filename)

        with gzip.GzipFile(
            filename="", fileobj=open(output_filename, "wb"), mode="wb", mtime=0
        ) as gzipped_tar, open(temp_unzipped_dir, "rb") as tar:
            gzipped_tar.write(tar.read())
    finally:
        os.remove(temp_unzipped_dir)
    os.chdir(current_dir)


class Env:
    def __init__(self):
        # simulate container env
        os.environ["SM_MODEL_DIR"] = f"{PROJECT_ROOT}/models/model_churn"
        os.environ["SM_CHANNEL_TRAIN"] = f"{PROJECT_ROOT}/datasets/churn"
        os.environ["SM_CHANNEL_TESTING"] = f"{PROJECT_ROOT}/datasets/churn"
        os.environ["SM_CHANNEL_VALIDATION"] = f"{PROJECT_ROOT}/datasets/churn"
        os.environ["SM_OUTPUT_DATA_DIR"] = f"{PROJECT_ROOT}/figures"


if __name__ == "__main__":
    import logging

    load_dotenv()
    log_level = int(os.getenv("LOG_LEVEL", logging.INFO))
    logger.setLevel(level=log_level)
    # Clean up metadata
    for dir in ["./mlruns"]:
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    Env()
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    prefix = "sagemaker/DEMO-xgboost-churn-pycharm/test_train_local/"

    args, unknown = _parse_args()
    parameters = vars(args)
    run_train_job(logger=logger, local=True, **parameters)
    archive_name = "model_local.tar.gz"
    make_model_tgzfile(
        output_filename=f"{PROJECT_ROOT}/models/{archive_name}",
        source_dir=os.environ["SM_MODEL_DIR"],
    )
    boto3.Session().resource("s3").Bucket(bucket).Object(
        os.path.join(prefix, archive_name)
    ).upload_file(os.path.join(PROJECT_ROOT, "models", archive_name))
