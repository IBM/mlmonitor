# SPDX-License-Identifier: Apache-2.0
import boto3
from botocore.exceptions import ClientError
import base64
import sys
import logging
import argparse
import json
import os
from pygit2 import Repository
from pygit2 import GitError
import pickle as pkl


def save_model(model, dir: str, name: str) -> tuple:
    """
    saves the sklearn model and xgboost model to a specified directory.
    The function returns the paths of both models.

    :param model: xgboost model to be save
    :param dir:str: Specify the directory where the model will be saved
    :param name:str: Name the model
    :return: A tuple of the sklearn model and the xgboost model
    """
    sk_model_location = os.path.join(dir, f"{name}-sk")
    bst_model_location = os.path.join(dir, f"{name}-bst")
    pkl.dump(model.model, open(sk_model_location, "wb"))
    model.model._Booster.save_model(bst_model_location)
    return sk_model_location, bst_model_location


def _parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    # CP4D specific arguments
    parser.add_argument("--catalog-id", type=str)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--model-entry-id", type=str)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--ibm-key-name", type=str, default="IBM_API_KEY_MLOPS")  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-env", type=str, default=os.getenv("ENV", "saas"), choices=["saas", "prem"], )  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-username", type=str, default=None)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-url", type=str, default=None)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--model-name", type=str, default="xgboost-model")

    # Training Job specific arguments (Sagemaker,Azure,WML) default SageMaker envar or Azure expected values
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./outputs"))
    parser.add_argument("--output-data-dir", type=str, default=os.getenv("SM_OUTPUT_DATA_DIR", "./outputs"))

    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TEST"))
    parser.add_argument("--validation", type=str, default=os.getenv("SM_CHANNEL_VALIDATION"))

    parser.add_argument("--region-name", type=str, default="ca-central-1")

    # XGBOOST specific hyperparameters
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--min-child-weight", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int, default=200)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--n-estimators", type=float, default=150)
    # fmt: on
    return parser.parse_known_args()


def git_branch(path: str, logger):
    try:
        repo = Repository(path)
        return repo.head.raw_shorthand.decode("utf-8")
    except GitError as e:
        logger.warning(f"GitError {e}")
        return "No Git repo found"


def _init_logger(level: int = logging.INFO) -> logging.Logger:
    """
    initializes the logger object for this module.
    It sets the log level to DEBUG and adds a StreamHandler to write output
    to sys.stdout

    :return: initialized logger instance
    """
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s churn module] : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def _get_secret(
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
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if "SecretString" in get_secret_value_response:
            return get_secret_value_response["SecretString"]
        else:
            return base64.b64decode(get_secret_value_response["SecretBinary"])


def read_columns():
    filepath = (
        "/opt/ml/code/model_signature.json"
        if os.path.exists("/opt/ml/code/model_signature.json")
        else os.path.join(os.path.dirname(__file__), "model_signature.json")
    )
    with open(filepath) as json_file:
        signature = json.load(json_file)

    return signature.get("signature").get("feature_columns")
