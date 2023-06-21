# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import time
import json
import requests
from collections import OrderedDict

from azureml.core.webservice import Webservice, AksWebservice, AciWebservice
from azureml.core import Workspace


def _score_unstructured(
    payload: np.array,
    workspace: Workspace,
    endpoint_name: str,
    content_type: str = "json",
):
    """
    Score an Azure ML Endpoint for unstructured dataset df (pandas dataframe): The dataframe to be scored.

    :param payload:np.array: Pass the unstructured data to be scored
    :param workspace:Workspace: AWS access key id
    :param endpoint_name:str: Specify the endpoint name to use
    :param content_type: Specify the type of data that is being sent to the endpoint
    :return: A list of predictions
    """
    service = Webservice(workspace=workspace, name=endpoint_name)
    if service.compute_type.lower() == "aks":
        service = AksWebservice(workspace=workspace, name=endpoint_name)
    elif service.compute_type.lower() == "aci":
        service = AciWebservice(workspace=workspace, name=endpoint_name)
    else:
        raise ValueError("compute_type should be aks or aci")

    start_time = time.time()

    if content_type != "json":
        raise ValueError("content_type provided should be csv or json")

    prediction = service.run(payload)

    response_time = int((time.time() - start_time) * 1000)
    print(f"response_time {response_time}ms")

    return prediction


def _score(
    df: pd.DataFrame,
    workspace: Workspace,
    endpoint_name: str,
    content_type: str = "json",
    compute_type: str = "aks",
    method: str = "requests",
    auth: bool = False,
) -> dict:
    """
    Score an Azure ML Endpoint for structured dataset df (pandas dataframe): The dataframe to be scored.
    :param df:pd.DataFrame: Pass the data to be scored
    :param workspace:Workspace: AWS access key id
    :param endpoint_name:str: Specify the endpoint name to use
    :param content_type: Specify the type of data that is being sent to the endpoint
    :param compute_type:str=aks:
    :param method:str=requests: method to use to score the model using Azure sdk or requests lib directly
    :param auth:bool=True: use authentication when method is set to requests
    :return: A list of predictions
    """
    if compute_type.lower() == "aks":
        service = AksWebservice(workspace=workspace, name=endpoint_name)
    elif compute_type.lower() == "aci":
        service = Webservice(workspace=workspace, name=endpoint_name)
    else:
        raise ValueError("compute_type should be aks or aci")

    start_time = time.time()

    if content_type != "json":
        raise ValueError("content_type provided should be csv or json")

    payload = {"input": df.to_dict(orient="records")}

    scoring_uri = service.scoring_uri

    if method == "requests":
        headers = {"Content-Type": "application/json"}
        if auth:
            key, _ = service.get_keys()
            headers["Authorization"] = f"Bearer {key}"

        resp = requests.post(scoring_uri, json=payload, headers=headers)
        if resp.status_code != 200:
            raise ValueError(f"status code != 200 {resp.status_code}")
        res_json = resp.json()

    elif method == "sdk":
        res_json = service.run(json.dumps(payload))
    else:
        raise ValueError(f"method {method} Not supported")

    if "error" in list(res_json.keys()):
        raise ValueError(f"Error from endpoint: {res_json}")

    response_time = int((time.time() - start_time) * 1000)
    print(f"response_time {response_time}ms")

    return res_json


def get_wos_response(
    df: pd.DataFrame,
    workspace: Workspace,
    endpoint_name: str,
    content_type: str = "json",
    compute_type: str = "aks",
    prediction_field: str = "Scored Labels",
    probability_field: str = "Scored Probabilities",
):
    scoring_response = _score(
        df=df,
        workspace=workspace,
        endpoint_name=endpoint_name,
        content_type=content_type,
        compute_type=compute_type,
    )

    if (
        return_fields := list(
            OrderedDict.fromkeys(
                [field for field in scoring_response.get("output") for field in field]
            )
        )
    ) == [prediction_field, probability_field]:

        return {
            "fields": return_fields,
            "values": [list(x.values()) for x in scoring_response.get("output")],
        }
    else:
        raise ValueError(
            f"return unique fields {return_fields} expecting {[prediction_field, probability_field]}"
        )


def get_scores_labels(
    df: pd.DataFrame,
    endpoint_name: str,
    workspace: Workspace,
):
    prediction_json = _score(
        df=df, workspace=workspace, endpoint_name=endpoint_name, content_type="json"
    )

    predicted_labels = np.array(
        [list(x.values())[0] for x in prediction_json.get("output")]
    )
    proba_scores = np.array(
        [list(x.values())[1] for x in prediction_json.get("output")]
    )
    proba_vectors = np.array(
        [[np.round(proba, 3), np.round(1 - proba, 3)] for proba in proba_scores]
    )
    # proba_vectors = [list(np.roll(np.array([np.round(proba, 3), np.round(1 - proba, 3)]), label))
    # for proba,label in zip(proba_scores, predicted_labels)]
    return proba_vectors, proba_scores, predicted_labels
