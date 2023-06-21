# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import numpy as np


def _score_unstructured(
    payload: np.array,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str,
    endpoint_name: str,
):
    # imports
    import boto3
    import time
    import json

    sm_runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    start_time = time.time()

    scoring_response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"input_data": [{"values": payload.tolist()}]}),
    )
    result = json.loads(scoring_response["Body"].read().decode())

    response_time = int((time.time() - start_time) * 1000)
    print(f"response_time {response_time}ms")

    return result


def _score(
    df: pd.DataFrame,
    access_id: str,
    secret_key: str,
    region: str,
    endpoint_name: str,
    content_type: str = "csv",
    subscription_id: str = None,
):
    """
    Score a sagemaker Endpoint for structured dataset df (pandas dataframe): The dataframe to be scored. access_id (
    str): AWS Access Key ID for the API call. secret_key (str): AWS Secret Access Key for the API call. region (str):
    Region of our SageMaker endpoint, e.g., 'us-east-2'. This is used to construct an instance of boto3's `Session`
    class and subsequently used by boto3's `client` method to create a connection with SageMaker endpoint using the
    specified endpoint_name

    :param df:pd.DataFrame: Pass the data to be scored
    :param access_id:str: AWS access key id
    :param secret_key:str: Access Key for AWS account
    :param region:str: Specify the region in which to run the inference
    :param endpoint_name:str: Specify the endpoint name to use
    :param content_type: Specify the type of data that is being sent to the endpoint
    :param subscription_id:str=None: Watson OpenScale subscription id
    :return: A list of predictions
    """
    # imports
    import boto3
    import time
    import json

    sm_runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        aws_access_key_id=access_id,
        aws_secret_access_key=secret_key,
    )
    start_time = time.time()

    if content_type == "csv":
        payload = "\n".join([",".join([str(x) for x in row]) for row in df.to_numpy()])
        content_type = "text/csv"
    elif content_type == "json":
        payload = {
            "instances": [{"features": features} for features in df.values.tolist()]
        }
        if subscription_id:
            payload["subscription_id"] = subscription_id
        content_type = "application/json"
        payload = json.dumps(payload)
    else:
        raise ValueError("content_type provided should be csv or json")

    res = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=payload
    )

    response_time = int((time.time() - start_time) * 1000)
    print(f"response_time {response_time}ms")

    model_output = res["Body"].read().decode("utf-8")
    # Extract the details
    pred_json = json.loads(model_output).get("predictions")

    return pred_json


def get_scores_labels(
    df: pd.DataFrame,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str,
    endpoint_name: str,
):
    prediction_json = _score(
        df=df,
        access_id=aws_access_key_id,
        secret_key=aws_secret_access_key,
        region=region_name,
        endpoint_name=endpoint_name,
        content_type="json",
    )

    predicted_labels = np.array([list(x.values())[0] for x in prediction_json])
    proba_scores = np.array([list(x.values())[1] for x in prediction_json])
    proba_vectors = np.array(
        [[np.round(proba, 3), np.round(1 - proba, 3)] for proba in proba_scores]
    )
    # proba_vectors = [list(np.roll(np.array([np.round(proba, 3), np.round(1 - proba, 3)]), label))
    # for proba,label in zip(proba_scores, predicted_labels)]
    return proba_vectors, proba_scores, predicted_labels


def get_wos_response(
    df: pd.DataFrame,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str,
    endpoint_name: str,
    prediction_field: str = "prediction",
    probability_field: str = "probability",
):
    return {
        "fields": [prediction_field, probability_field],
        "values": [
            list(x.values())
            for x in _score(
                df=df,
                access_id=aws_access_key_id,
                secret_key=aws_secret_access_key,
                region=region_name,
                endpoint_name=endpoint_name,
                content_type="json",
            )
        ],
    }


def sm_get_ep_configname(sagemaker_client, endpoint_name: str) -> str:
    ep = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    return ep.get("EndpointConfigName")


def sm_get_modelnames(sagemaker_client, endpoint_name: str) -> list:
    endpoint_cfg_name = sm_get_ep_configname(
        sagemaker_client=sagemaker_client, endpoint_name=endpoint_name
    )
    epcfg = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_cfg_name
    )
    return [x.get("ModelName") for x in epcfg.get("ProductionVariants")]


def delete_deployment_resources(sagemaker_client, endpoint_name: str) -> int:
    count = 0

    filtered_deployments = [
        resource
        for resource in sagemaker_client.list_endpoints(MaxResults=100).get("Endpoints")
        if resource.get("EndpointName") == endpoint_name
    ]

    if len(filtered_deployments) == 1:

        endpoint_cfg_name = sm_get_ep_configname(
            sagemaker_client=sagemaker_client, endpoint_name=endpoint_name
        )
        model_names = sm_get_modelnames(
            sagemaker_client=sagemaker_client, endpoint_name=endpoint_name
        )

        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_cfg_name)
        for model in model_names:
            count += 1
            sagemaker_client.delete_model(ModelName=model)

    return count
