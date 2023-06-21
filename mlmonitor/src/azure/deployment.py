# SPDX-License-Identifier: Apache-2.0
import logging
import hashlib

from typing import Optional, List, Union, Tuple, Dict
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.webservice import (
    Webservice,
    AksWebservice,
    AciWebservice,
    LocalWebservice,
)
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.exceptions import WebserviceException


def get_workspace(
    auth: ServicePrincipalAuthentication,
    workspace_name: str,
    az_rg: str,
    az_sub_id: str,
    az_location: str = "canadaeast",
) -> Workspace:
    """Creates or Gets an Azure Machine Learning Workspace

    :param auth:
    :param az_sub_id: str
    :param workspace_name: Name of the Workspace to create or fetch
    :param az_rg:str
    :param az_location:=canadaeast
    :return: Azure ML Workspace
    """

    workspace = Workspace.create(
        name=workspace_name,
        location=az_location,
        resource_group=az_rg,
        subscription_id=az_sub_id,
        auth=auth,
        exist_ok=True,
    )

    return workspace


def deploy_az_model(
    workspace: Workspace,
    model: Model,
    entry_script: str,
    environ_name: str,
    deployment_name: str,
    conda_packages: list,
    compute_type: str = "aks",
    redeploy: bool = False,
    cluster_name: Optional[str] = None,
    auth_enabled: bool = False,
    deploy_config_params: Optional[Dict] = None,
) -> Union[AksWebservice, AciWebservice]:
    """Deploy the given model from to AKS or ACI depending on compute_type parameter

    :param workspace:Workspace
    :param model: Model
    :param entry_script:str entry point for inference
    :param environ_name:str Environment name
    :param deployment_name:str deployment_name
    :param conda_packages:
    :param compute_type:str=aks
    :param redeploy:bool redeploy service if exists
    :param auth_enabled:bool Activate authentication
    :param cluster_name:Optional[str] Cluster name
    :param deploy_config_params:Optional[Dict] deployment config parameters

    :return: webservice object
    """

    # Prepare Inference Config ,Compute Target and Deployment Config
    inf_config = define_configuration(
        environ_name=environ_name,
        conda_packages=conda_packages,
        entry_script=entry_script,
    )

    if not deploy_config_params:
        deploy_config_params = {
            "cpu_cores": 1,
            "memory_gb": 1,
            "tags": {"source": "mlmonitor"},
            "description": "model use from mlmonitor case monitored in WOS",
        }

    if compute_type.lower() == "aks":
        # token_auth_enabled
        compute_target = create_aks_cluster(
            workspace=workspace, cluster_name=cluster_name
        )
        deploy_config_params["auth_enabled"] = False
        deploy_config_params["token_auth_enabled"] = auth_enabled
        deploy_config = AksWebservice.deploy_configuration(**deploy_config_params)

    elif compute_type.lower() == "aci":
        compute_target = None
        deploy_config_params["auth_enabled"] = auth_enabled
        deploy_config = AciWebservice.deploy_configuration(**deploy_config_params)
    elif compute_type.lower() == "local":
        compute_target = None
        deploy_config = LocalWebservice.deploy_configuration(port=8890)
    else:
        raise ValueError("compute_type should be aks aci or local")

    # Deploy or Redeploy Webservice
    deployments = get_deployments(
        deployment_name=deployment_name, workspace=workspace, compute_type=compute_type
    )

    if (num_deploy := len(deployments)) >= 1:
        raise ValueError(
            f"{len(deployments)} services found with name {deployment_name}"
        )

    if num_deploy == 0 or (num_deploy == 1 and redeploy):

        web_service = Model.deploy(
            workspace=workspace,
            name=deployment_name,
            models=[model],
            inference_config=inf_config,
            deployment_config=deploy_config,
            deployment_target=compute_target,
            overwrite=True,
        )

        web_service.wait_for_deployment(show_output=True)
        return web_service
    else:
        return deployments[0]


def create_aks_cluster(
    workspace: Workspace, cluster_name: str, create: bool = False
) -> ComputeTarget:
    """Creates Azure Kubernetes Service cluster in the Workspace

    :param workspace: Workspace client
    :param cluster_name: Cluster Name given or retrieved
    :param create: Create AKS cluster if provided cluster_name name does not exists
    :return: bool:False AKS Compute Target
    """

    try:
        aks_target = ComputeTarget(workspace=workspace, name=cluster_name)

    except ComputeTargetException as e:
        if not create:
            raise e

        prov_config = AksCompute.provisioning_configuration()

        aks_target = ComputeTarget.create(
            workspace=workspace,
            name=cluster_name,
            provisioning_configuration=prov_config,
        )

        if aks_target.get_status() != "Succeeded":
            aks_target.wait_for_completion(show_output=True)

    return aks_target


def register_model(
    workspace: Workspace,
    model_path: str,
    model_name: str,
    tags: dict,
    description: str = "Training from mlmonitor",
) -> Model:
    """Register model to AzureML Workspace

    :param description:
    :param tags:
    :param workspace: Azure Workspace client
    :param model_path: Local file path to upload to Azure ML
    :param model_name: Name given in Azure ML Workspace
    :return: Registered Model object
    """

    model = Model.register(
        model_path=model_path,  # Local file to upload and register as a model.
        model_name=model_name,  # Model name registered in Workspace
        tags=tags,
        description=description,
        workspace=workspace,
    )
    return model


def define_configuration(
    environ_name: str,
    conda_packages: list,
    entry_script: str,
    python_version: str = "3.9",
) -> InferenceConfig:
    """Define the inference configuration for the AKS webservice

    :param environ_name: Environment name
    :param conda_packages:
    :param entry_script:
    :param python_version:
    :return: Inference Config object
    """

    conda_deps = CondaDependencies.create(conda_packages=conda_packages)
    conda_deps.set_python_version(python_version)

    myenv = Environment(name=environ_name)
    myenv.python.conda_dependencies = conda_deps

    return InferenceConfig(
        entry_script=entry_script,
        environment=myenv,
    )


def is_deployed(
    deployment_name: str, workspace: Workspace, compute_type: Optional[str] = None
) -> bool:
    """
    checks if a training job with the specified name exists.
    It returns True if it does and was completed and False otherwise.

    :param workspace:Workspace Azure workspace Object
    :param deployment_name:str: Deployment name to be checked
    :param compute_type:str=None compute type for deployment_name
    :return: A boolean value indicating whether the deployment_name is deployed in azure ML
    """
    return (
        len(
            get_deployments(
                deployment_name=deployment_name,
                workspace=workspace,
                compute_type=compute_type,
            )
        )
        == 1
    )


def get_deployments(
    deployment_name: str, workspace: Workspace, compute_type: Optional[str] = None
) -> List:
    """
    get all endpoint deployed in Azure workspace

    :param workspace:Workspace Azure workspace Object
    :param deployment_name:str: Deployment name to be checked
    :param compute_type:str=None compute type for deployment_name
    :return: list of WebService objects
    """

    endpoints = Webservice.list(workspace=workspace)

    if compute_type:
        assert compute_type.lower() in {
            "aks",
            "aci",
        }, "compute_type should be aks or aci"
        return [
            ep
            for ep in endpoints
            if ep.name == deployment_name
            and ep.compute_type.lower() == compute_type.lower()
        ]
    else:
        return [ep for ep in endpoints if ep.name == deployment_name]


def get_deployment_models(deployment_name: str, workspace: Workspace) -> List:
    """
    get all models in Azure workspace for a given deployment name

    :param workspace:Workspace Azure workspace Object
    :param deployment_name:str: Deployment name to be checked
    :return: list of Models objects for deployment_name
    """
    return Webservice(workspace=workspace, name=deployment_name).models


def get_model_uid_by_name(
    model_name: str, workspace: Workspace, key: Optional[str] = "id"
) -> Optional[Union[Model, str]]:
    """
    returns the Azure model id for a given model name.
    searching through all models and returning the latest version.
    If no matching deployments are found, it will return None.

    :param model_name:str: model name in WML
    :param workspace: Azure Worksapce Object
    :param key:Optional[str]="id"  specific key to return from model resource details, return all resource if None
    :return: The Model object for given model_name with latest version
    """
    try:
        model = Model(workspace=workspace, name=model_name)
        return model.__dict__.get(key) if key in list(model.__dict__.keys()) else model
    except WebserviceException:
        return None


def get_deploy_facts(
    deployment_name: str, workspace: Workspace
) -> Tuple[Dict, Dict, Dict]:
    """
    collect deployment metadata for Azure Endpoint

    :param workspace:Workspace Azure workspace Object
    :param deployment_name:str: Deployment name to be checked
    :return: list of Models objects for deployment_name
    """

    ws = Webservice(workspace=workspace, name=deployment_name)

    model = ws.models[0]

    params = {
        # "entrypoint": inf_config.entry_script,
        # "compute_name": ws.compute_name,
        "compute_type": ws.compute_type,
    }
    metrics = {"created_time": ws.created_time.strftime("%m/%d/%Y %H:%M:%S")}
    tags = {
        "scoring_url": ws.scoring_uri,
        "model_id": model.id,
        # "sklearn_version": sklearn_version,
    }

    return metrics, params, tags


def delete_az_resources(
    deployment_name: str, workspace: Workspace, logger: Optional[logging.Logger] = None
) -> int:
    """
    Deletes all Azure resources associated with a deployment.

    :param deployment_name: str: name of the deployment
    :param workspace: Workspace: Azure workspace that contains the model
    :param logger: Optional[logging.Logger]: Log messages if specified
    :return: The number of deleted deployments
    """
    deployments = get_deployments(workspace=workspace, deployment_name=deployment_name)
    az_models = Model.list(workspace=workspace, name=deployment_name)
    if len(deployments) > 0:
        for deployment in deployments:
            az_models += deployment.models
            if logger:
                logger.info(f"Delete Azure endpoint endpoint {deployment_name}")
            deployment.delete()

    for model in az_models:
        if logger:
            logger.info(f"Delete Azure model {model.id}")
        model.delete()

    return len(deployments)


def list_deployment_assets_wos(deployment_name: str, workspace: Workspace):
    ws = Webservice(workspace=workspace, name=deployment_name)
    scoring_url = ws.scoring_uri
    created_at = ws.created_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return {
        "guid": hashlib.md5(scoring_url.encode()).hexdigest(),
        "url": scoring_url,
        "created_at": created_at,
        "modified_at": created_at,
        "entity": {
            "name": deployment_name,
            "deployment_rn": deployment_name,
            "type": "online",
            "description": "deployed by mlmonitor on Azure",
            "scoring_endpoint": {
                "url": scoring_url,
                "request_headers": {"Content-Type": "application/json; charset=UTF-8"},
            },
            "asset": {
                "asset_id": hashlib.md5(deployment_name.encode()).hexdigest(),
                "asset_rn": deployment_name,
                "url": scoring_url,
                "name": deployment_name,
                "asset_type": "model",
                "created_at": created_at,
            },
        },
        "metadata": {
            "guid": hashlib.md5(deployment_name.encode("utf-8")).hexdigest(),
            "url": scoring_url,
        },
    }
