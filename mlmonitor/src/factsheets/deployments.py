# SPDX-License-Identifier: Apache-2.0
import sagemaker
from boto3.session import Session
from typing import Optional


def add_aws_deployment_details(
    apikey: str,
    session: Session,
    endpoint_name: str,
    catalog_id: str = None,
    model_entry_id: str = None,
    inference_entrypoint: str = None,
    source_dir: str = None,
    description: str = "",
    framework_version: str = "unknown",
    framework: str = "unknown",
    env: str = "saas",
    cp4d_username: Optional[str] = None,
    cp4d_url: Optional[str] = None,
):
    from ibm_aigov_facts_client import (
        AIGovFactsClient,
        DeploymentDetails,
        CloudPakforDataConfig,
    )

    sagemaker_session = sagemaker.Session(session)
    endpoint_data = sagemaker_session.sagemaker_client.describe_endpoint(
        EndpointName=endpoint_name
    )
    endpoint_config_data = sagemaker_session.sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_data.get("EndpointConfigName")
    )
    model_data = sagemaker_session.sagemaker_client.describe_model(
        ModelName=endpoint_config_data.get("ProductionVariants")[0].get("ModelName")
    )

    if env == "prem":
        facts_client = AIGovFactsClient(
            cloud_pak_for_data_configs=CloudPakforDataConfig(
                service_url=cp4d_url,
                username=cp4d_username,
                api_key=apikey,
            ),
            experiment_name=endpoint_name,
            set_as_current_experiment=True,
            external_model=True,
            enable_autolog=False,
        )

    elif env == "saas":
        facts_client = AIGovFactsClient(
            api_key=apikey,
            experiment_name=endpoint_name,
            set_as_current_experiment=True,
            external_model=True,
            enable_autolog=False,
        )

    facts_client.manual_log.start_trace()
    # get Run ID
    run_id = facts_client.runs.get_current_run_id()
    print(f"Current RunID {run_id}")
    facts_client.manual_log.log_params({"EndpointName": endpoint_name})
    facts_client.manual_log.log_params(
        {
            "EndpointName": endpoint_data.get("EndpointName"),
            "EndpointArn": endpoint_data.get("EndpointArn"),
            "EndpointConfigName": endpoint_data.get("EndpointConfigName"),
            "CreationTime": endpoint_data.get("CreationTime").strftime(
                "%m/%d/%Y, %H:%M:%S"
            ),
            "EndpointStatus": endpoint_data.get("EndpointStatus"),
            "ModelName": model_data.get("ModelName"),
            "PrimaryContainer": model_data.get("PrimaryContainer").get("Image"),
            "ModelDataUrl": model_data.get("PrimaryContainer").get("ModelDataUrl"),
            "entry_point": inference_entrypoint,
            "source_dir": source_dir,
            "framework_version": framework_version,
            "framework": framework,
        }
    )

    deployment = DeploymentDetails(
        identifier=endpoint_name,
        name=endpoint_name,
        deployment_type="online",
        scoring_endpoint=endpoint_name,
    )

    facts_client.export_facts.export_payload_manual(run_id)

    fs_model = facts_client.external_model_facts.save_external_model_asset(
        model_identifier=model_data.get("ModelName"),
        name=model_data.get("ModelName"),
        description=description,
        deployment_details=deployment,
        catalog_id=catalog_id,
    )

    muc_utilities = facts_client.assets.get_ai_usecase(
        ai_usecase_id=model_entry_id,
        catalog_id=catalog_id,
    )

    fs_model.track(
        usecase=muc_utilities,
        approach=muc_utilities.get_approaches()[0],
        version_number="minor",  # "0.1.0"
    )
