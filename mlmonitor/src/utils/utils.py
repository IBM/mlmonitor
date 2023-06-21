# SPDX-License-Identifier: Apache-2.0
import argparse
import numpy as np
from datetime import date, datetime
import requests
from requests.auth import HTTPBasicAuth


def json_serialize(obj):
    """
    helper function that converts datetime objects to strings.
    It's used in the json_encode function below.

    :param obj: Specify the object to be serialized
    :return: A string
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def parse_args():
    """
    parses the arguments passed to the script.
    It returns a Namespace object containing all of the arguments and their values.
    The returned object can then be used as an input to other functions in this module.

    :return: A namespace object containing the arguments and their values
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=str,
        default="use_case_churn",
        metavar="sourcedir",
        help="Model Use case Source directory (python module)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="scikit",
        choices=["scikit", "xgboost"],
        metavar="MODEL",
        help="model type to be deployed",
    )
    parser.add_argument(
        "--train-entrypoint",
        type=str,
        default="train_cc_xg_boost.py",
        metavar="TRAIN",
        help="entrypoint for training",
    )
    parser.add_argument(
        "--inference-entrypoint",
        type=str,
        default="inference_cc_xg_boost.py",
        metavar="INFERENCE",
        help="entrypoint for inference",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sagemaker/DEMO-xgboost-churn",
        metavar="prefix",
        help="prefix for model path",
    )
    parser.add_argument(
        "--model-data",
        type=str,
        metavar="MODELDATA",
        help="model path in s3 for the new AWS deployment",
    )
    parser.add_argument(
        "--deployment-name",
        type=str,
        metavar="DEPLOYNAME",
        help="Deployment name for AWS deployment",
    )
    parser.add_argument(
        "--deployment-target",
        type=str,
        metavar="DEPLOYTARGET",
        default="aws",
        choices=["aws", "azure", "wml", "custom"],
        help="Deployment target environment",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        metavar="MODELNAME",
        help="Model name for AWS deployment",
    )
    parser.add_argument("--model-id", type=str, metavar="MODELID", help="Model id")
    parser.add_argument("--catalog-id", type=str, metavar="CATID", help="WKC Catalog")
    parser.add_argument(
        "--model-entry-id",
        type=str,
        metavar="MODELENTRYID",
        help="WKC Model use case ID",
    )
    parser.add_argument(
        "--deployment-space", type=str, metavar="DS", help="WML deployment space"
    )
    parser.add_argument(
        "--feedback-type",
        type=str,
        metavar="FEEDBACK",
        default="scoring",
        choices=["scoring", "no_scoring"],
        help="feedback logging type with scoring or without scoring request",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.m4.xlarge",
        metavar="INSTANCE",
        help="Endpoint name for AWS deployment",
    )
    parser.add_argument(
        "--inference-samples",
        type=int,
        default=2,
        metavar="NSAMPLES",
        help="Number of samples to be sent for inference",
    )
    parser.add_argument("--ibm-key-name", type=str, default="IBM_API_KEY_MLOPS")
    parser.add_argument("--custom", action="store_true", default=False)
    parser.add_argument("--deploy", action="store_true", default=False)
    # WOS specific
    parser.add_argument(
        "--wos-provider-name",
        type=str,
        metavar="SPNAME",
        default="AWS_Provider_Prod",
        help="WOS Service provider name",
    )
    parser.add_argument(
        "--wos-deployment-type",
        type=str,
        metavar="WOSDTYPE",
        choices=["production", "pre_production"],
        help="Deployment type in openscale",
    )
    parser.add_argument(
        "--wos-provider-description",
        type=str,
        metavar="SPDESCR",
        default="AWS Model monitoring",
        help="WOS Service provider description",
    )
    parser.add_argument("--wos-provider-keep", action="store_true", default=True)
    parser.add_argument("--wos-evaluate-quality", action="store_true", default=False)
    parser.add_argument("--wos-evaluate-fairness", action="store_true", default=False)
    parser.add_argument("--wos-evaluate-explain", action="store_true", default=False)
    parser.add_argument("--wos-evaluate-drift", action="store_true", default=False)
    parser.add_argument("--wos-evaluate-mrm", action="store_true", default=False)
    parser.add_argument("--keep-wos-monitor", action="store_true", default=False)
    return parser.parse_args()


def predict(data, predictor, rows=500):
    """
    takes a dataframe and returns predictions for each row in the dataframe.
    The predictor is an object of type sagemaker.predictor.RealTimePredictor, which is used to send data
    to an endpoint and get back predictions in csv format

    :param data: data to be predicted
    :param predictor: Sagemaker predictor object
    :param rows=500: Split the data into smaller chunks so that we don't send too much data to the endpoint at once
    :return: A list of predictions, one for each row in the input data
    """
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ""
    for array in split_array:
        predictions = "".join([predictions, predictor.predict(array).decode("utf-8")])

    return predictions.split("\n")[:-1]


def complete_dag(
    scoring_url: str,
    username: str,
    password: str,
    request_headers: dict,
    request_body: dict,
) -> dict:
    """
    REST call used to notify MLOPs Orchestrator backend that CP4D notebook job was completed
    used by mlmonitor-cp4d-ai-pipelines.ipynb

    :param scoring_url:str: url of the MLOPs Orchestrator service (backend)
    :param username:str: basic auth username
    :param password:str: basic auth password
    :param request_headers:dict: header information to the scoring endpoint
    :param request_body:dict: Pass the pipeline definition to the scoring service
    :return: A dictionary with the status code of the response from the api call
    """
    scoring_response = requests.put(
        f"{scoring_url}/api/pipelines",
        json=request_body,
        headers=request_headers,
        auth=HTTPBasicAuth(username, password),
        verify=False,
    )
    print(scoring_response.status_code)
    if scoring_response.status_code in [200, 201, 204]:
        return scoring_response.json()
    else:
        return {"status_code": scoring_response.status_code}


def trigger_dag(url: str, username: str, password: str, request_body: dict) -> dict:
    """
    REST call used to trigger CP4D notebook job from in MLOPs Orchestrator backend
    used by MLOPs orchestrator frontend

    :param scoring_url:str: url of the MLOPs Orchestrator service (backend)
    :param username:str: basic auth username
    :param password:str: basic auth password
    :param request_body:dict: Pass the pipeline definition to the scoring service
    :return: A dictionary with the status code of the response from the api call
    """
    request_headers = {"Content-Type": "application/json"}

    scoring_response = requests.post(
        f"{url}/api/pipelines",
        json=request_body,
        headers=request_headers,
        auth=HTTPBasicAuth(username, password),
        verify=False,
    )
    print(scoring_response.status_code)
    if scoring_response.status_code in [200, 201, 204]:
        return scoring_response.json()
    else:
        return {"status_code": scoring_response.status_code}


def send_alert(url: str, username: str, password: str, request_body: dict) -> dict:
    """
    REST call used to notify MLOPs Orchestrator of Alerts
    Used by automation_alerts.ipynb

    :param url:str: url of the MLOPs Orchestrator service (backend)
    :param username:str: basic auth username
    :param password:str: basic auth password
    :param request_body:dict: Pass the pipeline definition to the scoring service
    :return: A dictionary with the status code of the response from the api call
    """
    request_headers = {"Content-Type": "application/json"}

    scoring_response = requests.post(
        f"{url}/api/alerts",
        json=request_body,
        headers=request_headers,
        auth=HTTPBasicAuth(username, password),
        verify=False,
    )
    print(scoring_response.status_code)
    if scoring_response.status_code in [200, 201, 204]:
        return scoring_response.json()
    else:
        return {"status_code": scoring_response.status_code}
