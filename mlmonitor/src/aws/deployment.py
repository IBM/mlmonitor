# SPDX-License-Identifier: Apache-2.0
import os
from sagemaker import Session as sm_Session
from sagemaker import image_uris
from mlmonitor.src import PROJECT_ROOT, ROLE
from typing import Dict, Optional
from boto3.session import Session as boto_Session


def generate_base_deployment_params(
    trained_model_data: str,
    source_dir: str,
    framework: str,
    framework_version: str,
    py_version: str,
    script: str,
    instance: str,
) -> Dict:
    """
    generates the base parameters for deploying am AWS Sagemaker Endpoint deployment.
    It takes in 7 arguments: trained_model_data, source_dir, and runtime information.
    trained_model_data is the location of the trained model data on S3.
    source_dir is where to find your inference script on S3 (e.g., /opt/ml/code).
    runtime information such  as framework you are using (e.g., tensorflow),
    the version of that framework you're using (e.g., 1), and what instance type you want to use for inference.

    :param trained_model_data:str: Specify the location of the trained model artifact
    :param source_dir:str: Specify the directory in which the model scripts are located
    :param framework:str:
    :param framework_version:str:
    :param py_version:str:
    :param script:str:
    :param instance:str:
    :return: A dictionary of parameters that will be passed to the deploy_model function
    """

    assert framework in {"pytorch", "sklearn", "tensorflow", "xgboost"}

    container = image_uris.retrieve(
        framework=framework,
        region=os.environ.get("AWS_DEFAULT_REGION", "ca-central-1"),
        version=framework_version,
        image_scope="inference",
        instance_type=instance,
    )
    print(container)

    deployment_model_params = {
        "model_data": trained_model_data.strip(),
        "source_dir": f"{PROJECT_ROOT}/{source_dir}",
        "image_uri": container,
        "role": ROLE,
        "entry_point": script,
        "framework_version": framework_version,
        "container_log_level": 20,
    }

    if framework != "tensorflow":
        deployment_model_params["py_version"] = py_version

    if framework in {"tensorflow", "pytorch"}:
        deployment_model_params["container_log_level"] = (
            20,
        )  # 10 debug 20 info 30 warning 40 error

    return deployment_model_params


def is_deployed(
    deployment_name: str, sagemaker_client: Optional[boto_Session.client] = None
) -> bool:
    """
    checks if a Sagemaker endpoint with the specified name exists.
    It returns True if it does and was completed and False otherwise.

    :param sagemaker_client: Optional Sagemaker client to pass as argument if already instantiated
    :param deployment_name:str: Deployment name to be checked
    :return: A boolean value indicating whether the deployment_name is deployed in Sagemaker
    """
    if not sagemaker_client:
        session = boto_Session()
        sagemaker_client = session.client("sagemaker")

    filtered_deployments = [
        resource
        for resource in sagemaker_client.list_endpoints(MaxResults=100).get("Endpoints")
        if resource.get("EndpointName") == deployment_name
    ]
    return len(filtered_deployments) == 1


def describe_ep(deployment_name: str, **aws_credentials) -> Dict:
    """
    returns a dictionary containing the Sagemaker endpoint description

    :param deployment_name: str: endpoint name
    :param **aws_credentials: Pass in the aws_access_key_id, aws_secret_access key and region
    :return: A dictionary with EndpointArn CreationTime LastModifiedTime EndpointStatus
    """
    session = boto_Session(**aws_credentials)
    sagemaker_session = sm_Session(session)
    return sagemaker_session.sagemaker_client.describe_endpoint(
        EndpointName=deployment_name
    )
