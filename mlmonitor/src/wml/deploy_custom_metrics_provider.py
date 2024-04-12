# SPDX-License-Identifier: Apache-2.0
import ibm_watson_machine_learning
import json
from typing import Callable

from mlmonitor.src import logger
from mlmonitor.src.wml import wml_client, WML_SPACE_ID


def deploy_custom_metrics_provider(
    deployment_name: str,
    python_function_name: str,
    function_code: Callable,
    wml_client: ibm_watson_machine_learning.APIClient = wml_client,
    wml_space_id: str = WML_SPACE_ID,
    runtime: str = "runtime-23.1-py3.10",
    hardware_specifications: str = "S",
) -> str:
    """
    deploys a custom metrics provider to the WML instance.

    :param deployment_name:str: deployment Name
    :param python_function_name:str: Name of python function asset to associate with this deployment
    :param function_code:Callable: Pass the python function code
    :param wml_client:ibm_watson_machine_learning.APIClient=wml_client: Watson Machine learning API Client
    :param wml_space_id:str=WML_SPACE_ID: Watson Machine learning deployment space id
    :param runtime:str=&quot;runtime-22.1-py3.9&quot;: WML runtime of the deployment
    :param hardware_specifications:str=S: hardware type used to run the python function deployment
    :return: The scoring url of the deployed function
    """
    assert hardware_specifications in {"XL", "L", "M", "S", "XS", "XXS"}
    wml_client.set.default_space(wml_space_id)

    # Create the function meta properties.
    software_spec_id = wml_client.software_specifications.get_id_by_name(runtime)
    logger.info(software_spec_id)
    function_meta_props = {
        wml_client.repository.FunctionMetaNames.NAME: python_function_name,
        wml_client.deployments.ConfigurationMetaNames.TAGS: ["mlmonitor"],
        wml_client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: software_spec_id,
    }

    # Store the Python function.
    function_artifact = wml_client.repository.store_function(
        meta_props=function_meta_props, function=function_code
    )
    function_uid = wml_client.repository.get_function_id(function_artifact)
    logger.info(
        f"Function {python_function_name} created with Function UID = {function_uid}"
    )
    function_details = wml_client.repository.get_details(function_uid)
    logger.info(
        f"Function {python_function_name} Details:\n{json.dumps(function_details, indent=4)}"
    )

    # Deploy the Python function.
    hardware_spec_id = wml_client.hardware_specifications.get_id_by_name(
        hardware_specifications
    )

    # Create deployment metadata for the Python function.
    deploy_meta = {
        wml_client.deployments.ConfigurationMetaNames.NAME: deployment_name,
        wml_client.deployments.ConfigurationMetaNames.TAGS: ["mlmonitor"],
        wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
        wml_client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {
            "id": hardware_spec_id
        },
    }
    # Create a deployment.
    deployment_details = wml_client.deployments.create(
        function_uid, meta_props=deploy_meta
    )
    # Get the scoring URL.
    created_at = deployment_details["metadata"]["created_at"]
    find_string_pos = created_at.find("T")
    scoring_url = wml_client.deployments.get_scoring_href(deployment_details)
    if find_string_pos != -1:
        current_date = created_at[:find_string_pos]
        scoring_url = f"{scoring_url}?version={current_date}"

    return scoring_url
