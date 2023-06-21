# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Optional
import logging
from mlmonitor.config import get_azure_credentials
from mlmonitor.src import logger

try:
    from azureml.core.authentication import ServicePrincipalAuthentication
    from azureml.core import Workspace
except ModuleNotFoundError:
    logger.warning("run pip install mlmonitor[azure] to use AzureModelUseCase")

SUPPORTED_AZURE_COMPUTE = {"aci", "aks"}


def get_workspace_age(
    workspace: Workspace, log: Optional[logging.Logger] = None
) -> str:
    """
    returns the age of an Azure Machine Learning workspace.

    :param workspace: Workspace: workspace object
    :param log: Optional[logging.Logger]: Pass in a logger object
    :return: creation date of the workspace
    """
    creation_date = workspace.get_details().get("creationTime")
    ws_age = datetime.now() - datetime.strptime(
        workspace.get_details().get("creationTime")[:-2], "%Y-%m-%dT%H:%M:%S.%f"
    )
    if log:
        log.info(
            f"AZ ML workspace {workspace.name} in resource group {workspace.resource_group} fetched created {ws_age.days} "
            f"days {ws_age.seconds // 3600} hours {ws_age.seconds % 3600 // 60} "
            f"minutes {ws_age.seconds % 3600 % 60} seconds ago."
        )
    return creation_date


(
    az_ws_name,
    az_rg,
    az_sub_id,
    az_tenant_id,
    az_sp_id,
    az_sp_secret,
) = get_azure_credentials()

try:
    logger.debug(f"Instantiate AZURE ML Workspace {az_ws_name}")

    AZ_SP_AUTH = ServicePrincipalAuthentication(
        tenant_id=az_tenant_id,
        service_principal_id=az_sp_id,
        service_principal_password=az_sp_secret,
    )

    AZ_WORKSPACE = Workspace.get(
        name=az_ws_name,
        subscription_id=az_sub_id,
        resource_group=az_rg,
        auth=AZ_SP_AUTH,
    )

    get_workspace_age(workspace=AZ_WORKSPACE, log=logger)

except Exception as e:
    AZ_WORKSPACE = None
    logger.warning(f"AZ_WORKSPACE {az_ws_name} instantiation failed : {e}")
