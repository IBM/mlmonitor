# SPDX-License-Identifier: Apache-2.0
from ibm_watson_machine_learning import APIClient
from typing import Optional


def get_deployment_uid_by_name(
    wml_client: APIClient, deployment_name: str, key: Optional[str] = "id"
) -> Optional[str]:
    """
    returns the WML deployment uid for a given deployment name.
    searching through all deployments and returning the first one that matches the deployment_name.
    If no matching deployments are found, it will return None.

    :param wml_client:APIClient: Watson Machine Learning API Client
    :param deployment_name:str: deployment name in WML
    :return: The deployment_uid for a given deployment_name
    :param key:Optional[str]="id" specific key to return from deployment resource details, return all resource if None
    """
    all_deployment = wml_client.deployments.get_details()
    deployment_uids = [
        deploy.get("metadata").get(key) if key else deploy
        for deploy in all_deployment.get("resources")
        if deploy.get("entity").get("name") == deployment_name
    ]

    if len(deployment_uids) == 1:
        return deployment_uids[0]
    elif len(deployment_uids) == 0:
        return None
    else:
        raise ValueError(
            f"{len(deployment_uids)} deployments found for name {deployment_name} expecting 1 or 0"
        )


def get_model_uid_by_name(
    wml_client: APIClient, model_name: str, key: Optional[str] = "id"
) -> Optional[str]:
    """
    returns the WML model uid for a given model name.
    searching through all models and returning the first one that matches the model_name.
    If no matching deployments are found, it will return None.

    :param wml_client:APIClient: Watson Machine Learning API Client
    :param model_name:str: model name in WML
    :param key:Optional[str]="id"  specific key to return from model resource details, return all resource if None
    :return: The model_uid for a given model_name
    """
    all_models = wml_client.repository.get_model_details()
    model_uids = [
        model.get("metadata").get(key) if key else model
        for model in all_models.get("resources")
        if model.get("metadata").get("name") == model_name
    ]

    if len(model_uids) == 1:
        return model_uids[0]
    elif len(model_uids) == 0:
        return None
    else:
        raise ValueError(
            f"{len(model_uids)} models found for name {model_name} expecting 1 or 0"
        )


def get_function_uid_by_name(
    wml_client: APIClient, function_name: str, key: Optional[str] = "id"
) -> Optional[str]:
    """
    returns the unique identifier for a function with the given name.
    If no function is found, it returns None.
    If multiple functions are found, it raises an error.

    :param wml_client:APIClient: Watson Machine Learning API Client
    :param function_name:str: Specify the name of the function to be retrieved
    :param key:Optional[str]="id" specific key to return from function resource details, return all resource if None
    :return: The unique id of the function with the given name
    """
    all_functions = wml_client.repository.get_function_details()
    function_uids = [
        function.get("metadata").get(key) if key else function
        for function in all_functions.get("resources")
        if function.get("metadata").get("name") == function_name
    ]

    if len(function_uids) == 1:
        return function_uids[0]
    elif len(function_uids) == 0:
        return None
    else:
        raise ValueError(
            f"{len(function_uids)} functions found for name {function_name} expecting 1 or 0"
        )
