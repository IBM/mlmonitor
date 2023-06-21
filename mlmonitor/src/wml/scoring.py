# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import time
from ibm_watson_machine_learning import APIClient
from typing import Optional

from mlmonitor.src import logger
from mlmonitor.src.wml.utils import get_deployment_uid_by_name
from mlmonitor.src.wml import WML_SPACE_ID


def _score_unstructured(payload: np.array, endpoint_name: str, client: APIClient):
    """
     calls WML scoring endpoint of the deployed model.
     It accepts two parameters: payload and endpoint_name. The payload parameter contains data in a expected format
     while the endpoint_name parameter specifies which WML deployment name should be scored.
     The function returns predictions from WML deployment.

     :param payload:np.array: unstructured data image to be scored
     :param endpoint_name:str: Specify the name of the deployed WML endpoint
     :param client:APIClient: Watson Machine Learning Client
     :return: The predictions of the deployed model
     {'predictions':
     [
     {'id': 'dense_1',
    'fields': ['prediction', 'prediction_classes', 'probability'],
    'values': []}
     ]}

    """
    start_time = time.time()
    deployment_uid = get_deployment_uid_by_name(
        wml_client=client, deployment_name=endpoint_name
    )
    scoring_data = {"input_data": [{"values": [x.tolist() for x in payload]}]}

    predictions = client.deployments.score(deployment_uid, scoring_data)
    response_time = int((time.time() - start_time) * 1000)
    print(f"response_time {response_time}ms")

    return predictions


def _score(df: pd.DataFrame, endpoint_name: str, client: APIClient) -> dict:
    """
    Score a WML Endpoint for structured dataset df (pandas dataframe): The dataframe to be scored.
    WML client is used to send a scoring request to the specified endpoint_name following the fields,values format

    :param df:pd.DataFrame: Pass the data to be scored
    :param endpoint_name:str: S Specify the endpoint name to use
    :param client:APIClient: Access the watson machine learning api
    :return: A dictionary with two keys fields and values:
    """
    deployment_uid = get_deployment_uid_by_name(
        wml_client=client, deployment_name=endpoint_name
    )

    scoring_payload = {
        "input_data": [
            {
                "fields": df.columns.to_list(),
                "values": df.values.tolist(),
            }
        ]
    }
    logger.debug(f"Scoring request WML: {scoring_payload}")
    scoring_response = client.deployments.score(deployment_uid, scoring_payload)
    return scoring_response.get("predictions")[0]


def get_scores_labels(df: pd.DataFrame, endpoint_name: str, wml_client: APIClient):
    prediction_json = _score(df=df, endpoint_name=endpoint_name, client=wml_client)

    predicted_labels = [x[0] for x in prediction_json.get("values")]
    proba_scores = [x[1][0] for x in prediction_json.get("values")]
    proba_vectors = [x[1] for x in prediction_json.get("values")]
    return proba_vectors, proba_scores, predicted_labels


def get_wos_response(df: pd.DataFrame, endpoint_name: str, wml_client: APIClient):
    return _score(df=df, endpoint_name=endpoint_name, client=wml_client)


def delete_deployment_resources(wml_client, endpoint_name: str) -> int:
    """
    deletes the deployment resources associated with a given endpoint name.
    It returns the number of resources deleted.

    :param wml_client: Watson Machine Learning Client
    :param endpoint_name:str: Specify the name of the endpoint to delete
    :return: An integer value of the number of resources it deleted
    """
    count = 0
    if deployment_uid := get_deployment_uid_by_name(
        wml_client=wml_client, deployment_name=endpoint_name
    ):
        wml_client.deployments.delete(deployment_uid=deployment_uid)
        count += 1

    return count


def get_deployment_scoring_url(
    wml_client: APIClient, deployment_name: str
) -> Optional[str]:
    """
    returns the scoring URL for a given WML deployment name.

    :param wml_client:APIClient: Watson Machine Learning API Client
    :param deployment_name:str: deployment anme for which scoring url shouldbe returned
    :return: The scoring url for the deployment
    """
    all_deployment = wml_client.deployments.get_details()
    wml_client.set.default_space(WML_SPACE_ID)
    deployment_details = [
        deploy
        for deploy in all_deployment.get("resources")
        if deploy.get("entity").get("name") == deployment_name
    ]

    if len(deployment_details) == 1:
        deployment_details = deployment_details[0]
        scoring_url = wml_client.deployments.get_scoring_href(deployment_details)
        return scoring_url

    elif len(deployment_details) == 0:
        return None
    else:
        raise ValueError(
            f"{len(deployment_details)} deployments found for name {deployment_name} expecting 1 or 0"
        )
