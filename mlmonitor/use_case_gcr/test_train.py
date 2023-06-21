# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import shutil
import argparse

from use_case_gcr.train_gcr import (
    init_logger,
    init_external_fs_client,
    train,
    save_fs_model,
    fetch_dataset,
)
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
GCR_DIR = os.path.dirname(os.path.abspath(__file__))


class Env:
    def __init__(self):
        # simulate container env
        os.environ["SM_MODEL_DIR"] = f"{PROJECT_ROOT}/models/model_gcr"
        os.environ["SM_CHANNEL_TRAIN"] = f"{PROJECT_ROOT}/datasets/gcr"
        os.environ["SM_CHANNEL_TESTING"] = f"{PROJECT_ROOT}/datasets/gcr"


if __name__ == "__main__":
    load_dotenv()
    Env()

    for dir in ["./mlruns", "./logs"]:
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    parser = argparse.ArgumentParser()
    # fmt: off
    # CP4D specific arguments
    parser.add_argument("--catalog-id", type=str)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--model-entry-id", type=str)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--ibm-key-name", type=str, default="IBM_API_KEY_MLOPS")  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-env", type=str, default=os.getenv("ENV", "saas"), choices=["saas", "prem"], )  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-username", type=str, default=None)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-url", type=str, default=None)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--model-name", type=str, default="gcr-model")

    # Training Job specific arguments (Sagemaker,Azure,WML) default SageMaker envar or Azure expected values
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./outputs"))
    parser.add_argument("--output-data-dir", type=str, default=os.getenv("SM_OUTPUT_DATA_DIR", "./outputs"))

    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TEST"))
    parser.add_argument("--validation", type=str, default=os.getenv("SM_CHANNEL_VALIDATION"))

    parser.add_argument("--region-name", type=str, default="ca-central-1")
    # fmt: on

    log_level = int(os.getenv("LOG_LEVEL", logging.INFO))
    args = parser.parse_args()
    parameters = vars(args)
    logger = init_logger(level=log_level)
    (
        facts_client,
        props,
        EXPERIMENT_NAME,
        EXPERIMENT_ID,
        tags,
        params,
    ) = init_external_fs_client(logger=logger, **parameters)
    logger.debug(f"test_train parameters:\n{json.dumps(parameters,indent=4)}")

    logger.info(f"train {os.listdir(parameters.get('train'))}")
    train_data = fetch_dataset(data_path=parameters.get("train"))
    target_label_name = "Risk"
    features = [feat for feat in train_data.columns if feat != target_label_name]

    train(model_dir=parameters.get("model_dir"), logger=logger, train_data=train_data)

    save_fs_model(
        logger=logger,
        catalog_id=parameters.get("catalog_id"),
        model_entry_id=parameters.get("model_entry_id"),
        facts_client=facts_client,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        tags=tags,
        inputs=None,
        outputs=None,
        tdataref=None,
        params=params,
    )
