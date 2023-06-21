# SPDX-License-Identifier: Apache-2.0
import os

from ibm_watson_openscale.supporting_classes.enums import TargetTypes
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import Target

from mlmonitor.src import DATA_ROOT, API_KEY, IAM_URL, logger, ENV
from mlmonitor.src.utils.validation import is_csv
from mlmonitor.data import BUCKET_NAME, COS_ENDPOINT, COS_RESOURCE_CRN, CLOUD_API_KEY
from mlmonitor.data.cos import put_item
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.data_mart import create_datamart
from mlmonitor.src.wos.subscription import create_classification_subscription
from mlmonitor.src.wos.run_payload_logging import log_payload_data
from mlmonitor.src.wos.service_provider import (
    delete_provider,
    add_aws_provider,
    add_wml_provider,
    add_azure_provider,
)
from mlmonitor.src.model.config import ModelConfig

try:
    import azureml

    print(f"azureml at {azureml.__path__}")
    from mlmonitor.src.azure.deployment import list_deployment_assets_wos
except ModuleNotFoundError:
    print("run pip install mlmonitor[azure] to use AzureModelUseCase")


def monitor_model(
    model_config: ModelConfig,
    deployment_name: str,
    deployment_target: str,
    wos_provider_type: str,
    wos_provider_name: str,
    wos_provider_description: str,
    wos_provider_keep: bool = True,
    data_path: str = DATA_ROOT,
) -> str:
    """
    Configure a Watson OpenScale subscription for an online Endpoint name given by <deployment_name> in AWS Sagemaker.
    As a result of this action , the selected model should be Monitored for Quality and explainability with <wos_provider_name>
    as Machine Learning provider.

    :param model_config:ModelConfig: Sagemaker model config Object
    :param deployment_name:str: indicates Endpoint Name in Sagemaker for which an OpenScale subscription should be created
    :param deployment_target:str: target model serving environment (wml , aws , azure , custom)
    :param wos_provider_type:str: type of Watson OpenScale ML service provider to create if it does not already exists. options are `production` or `pre_production
    :param wos_provider_name:str: indicates OpenScale ML service provider unique name to use (if exists and wos_provider_keep=True) or create (if does not exist and wos_provider_keep=False)
    :param wos_provider_description:str: indicates description to add to OpenScale ML service provider if created (does not exists yet and wos_provider_keep=False)
    :param wos_provider_keep:bool=True: flag indicates whether an existing OpenScale ML service provider with the same name should be reused or re-created
    :param data_path:str=DATA_ROOT: location of dataset to get scoring request samples
    :param : Configure the mrm monitor
    :return: Watson OpenScale subscription id
    """

    assert deployment_target in {
        "wml",
        "aws",
        "azure",
    }, "Deployment target should be wml , aws or azure"

    ######################################################################
    # RETRIEVE MODEL USE CASE SPECIFIC CONFIGURATION DETAILS (SIGNATURE) #
    ######################################################################

    mrm_monitor = model_config.mrm_monitor_enabled

    training_data_file_name = model_config.training_data

    features = model_config.feature_columns
    target = model_config.class_label
    train_data_path = os.path.join(data_path, model_config.training_data)
    dataset_is_csv = is_csv(train_data_path)

    ###########################################################
    # HANDLING TRAINING DATA REFERENCE  FOR WOS SUBSCRIPTION  #
    ############################################################

    if (data_type := model_config.data_type) == "structured" and dataset_is_csv:

        put_item(item_name=training_data_file_name, item_path=data_path)

    elif data_type == "unstructured_image":
        training_data_file_name = None

    else:
        raise ValueError("data_type should be structured (csv) or unstructured_image ")

    assert deployment_target in {
        "wml",
        "aws",
        "azure",
    }, "deployment_target should be aws ,wml or azure"

    logger.debug(
        f"Deployment Name {deployment_name}\n"
        f"Deployment Target {deployment_target}\n"
        f"Provider Name [{wos_provider_name}] to create\n"
        f"WOS provider type [{wos_provider_type}]"
        f"Number features [{len(features) if features else 0}]"
        f"target [{target}]"
    )

    if not wos_provider_keep:
        delete_provider(wos_client=wos_client, service_provider_name=wos_provider_name)

    if deployment_target == "aws":
        from mlmonitor.src import aws_credentials as AWS_CREDENTIALS

        service_provider_id = add_aws_provider(
            wos_client=wos_client,
            wos_provider_type=wos_provider_type,
            service_provider_name=wos_provider_name,
            service_provider_descr=wos_provider_description,
            access_key_id=AWS_CREDENTIALS.get("aws_access_key_id"),
            secret_access_key=AWS_CREDENTIALS.get("aws_secret_access_key"),
            region=AWS_CREDENTIALS.get("region_name"),
        )
        deployment_space_id = None

    elif deployment_target == "azure":
        from mlmonitor.src.azure import (
            az_sp_id,
            az_sub_id,
            az_sp_secret,
            az_tenant_id,
            AZ_WORKSPACE,
        )

        service_provider_id = add_azure_provider(
            wos_client=wos_client,
            wos_provider_type=wos_provider_type,
            service_provider_name=wos_provider_name,
            service_provider_descr=wos_provider_description,
            client_id=az_sp_id,
            client_secret=az_sp_secret,
            subscription_id=az_sub_id,
            tenant_id=az_tenant_id,
        )
        deployment_space_id = None

    elif deployment_target == "wml":
        from mlmonitor.src.wml import (
            # wml_client as WML_CLIENT,
            WML_SPACE_ID,
            WML_URL,
        )

        service_provider_id = add_wml_provider(
            wos_client=wos_client,
            wos_provider_type=wos_provider_type,
            service_provider_name=wos_provider_name,
            service_provider_descr=wos_provider_description,
            apikey=API_KEY,
            url=WML_URL,
            space_id=WML_SPACE_ID,
        )
        deployment_space_id = WML_SPACE_ID

    else:
        raise NotImplementedError(
            "Only 'wml' ,'aws' or 'azure' service providers are supported"
        )

    logger.info(
        f"Service provider {wos_provider_name} created with Provider id {service_provider_id}"
    )

    data_mart_id = create_datamart(wos_client=wos_client)

    # Fails for Azure ML Service Providers
    # TODO verify wos_client fix list_assets
    if deployment_target != "azure":
        asset_deployment_details = wos_client.service_providers.list_assets(
            data_mart_id=data_mart_id,
            service_provider_id=service_provider_id,
            deployment_space_id=deployment_space_id,
        ).result

        assert (
            len(asset_deployment_details) > 0
        ), f"No deployments found for ML provider {wos_provider_name}"

        deployment_kv = {
            x.get("entity").get("name"): x.get("metadata").get("guid")
            for x in asset_deployment_details.get("resources")
        }

        deployment_uid = deployment_kv.get(deployment_name)
        assert (
            deployment_uid
        ), f"No deployments id found for {deployment_name} on ML provider {wos_provider_name}"
        logger.info(
            f'{deployment_target} deployment "{deployment_name}"  exists with DEPLOYMENT_ID : {deployment_uid}'
        )

        model_asset_details = wos_client.service_providers.get_deployment_asset(
            data_mart_id=data_mart_id,
            service_provider_id=service_provider_id,
            deployment_id=deployment_uid,
            deployment_space_id=deployment_space_id,
        )

    else:
        model_asset_details = list_deployment_assets_wos(
            deployment_name=deployment_name, workspace=AZ_WORKSPACE
        )
        deployment_uid = model_asset_details.get("metadata").get("guid")

    subscriptions = wos_client.subscriptions.list().result.subscriptions

    for subscription in subscriptions:
        sub_model_id = subscription.entity.asset.asset_id
        if sub_model_id == deployment_uid:
            wos_client.subscriptions.delete(subscription.metadata.id)
            logger.info(f"Deleted existing subscription for model {deployment_uid}")

    if deployment_target == "aws":
        model_asset_details = model_asset_details["resources"][0]

    # !TODO ask #openScale-support why prediction_field should be unset for wml : multiclass classification ?
    # explainability not working for WML on unstructured_image if prediction_field is set  `scoring_prediction` field expected
    # prediction array , probability array , scoring_prediction integer , prediction_probability double , scoring_input binary
    prediction_field = (
        None
        if deployment_target == "wml" and data_type == "unstructured_image"
        else model_config.prediction_field
    )

    logger.info(model_asset_details)

    ##############################################################
    # CONFIGURE WATSON OPENSCALE SUBSCRIPTION BASED ON SIGNATURE #
    ##############################################################
    if (problem_type := model_config.problem_type) in ["binary", "multiclass"]:
        subscription_id = create_classification_subscription(
            wos_client=wos_client,
            service_provider_id=service_provider_id,
            data_mart_id=data_mart_id,
            model_asset_details=model_asset_details,
            bucket_name=BUCKET_NAME,
            cos_resource_crn=COS_RESOURCE_CRN,
            cos_endpoint=COS_ENDPOINT,
            apikey=CLOUD_API_KEY,
            iam_url=IAM_URL,
            features=features,
            categorical_fields=model_config.categorical_columns,
            target=target,
            prediction_field=prediction_field,
            probability_fields=model_config.probability_fields,
            training_data_file_name=training_data_file_name,
            classification_type=problem_type,
            data_type=data_type,
            env=ENV,
        )
    else:
        raise ValueError(f"{model_config.problem_type} problem type not implemented")
    ###################
    # PAYLOAD LOGGING #
    ###################
    payload_res = log_payload_data(
        model_config=model_config,
        deployment_name=deployment_name,
        deployment_target=deployment_target,
        inference_samples=2,
    )
    assert payload_res.get(
        "success"
    ), f"Payload Logging Failed for new subscription {subscription_id}"
    #########################
    # CONFIGURE MRM MONITOR #
    #########################
    if mrm_monitor:
        # Enable MRM Monitor
        mrm_monitor_details = wos_client.monitor_instances.create(
            data_mart_id=data_mart_id,
            background_mode=False,
            monitor_definition_id=wos_client.monitor_definitions.MONITORS.MODEL_RISK_MANAGEMENT_MONITORING.ID,
            target=Target(
                target_type=TargetTypes.SUBSCRIPTION, target_id=subscription_id
            ),
            parameters={},
        ).result

        mrm_instance_id = mrm_monitor_details.metadata.id
        logger.info(f"MRM Monitor ID Created [{mrm_instance_id}]")

    wos_client.data_sets.print_records_schema(
        data_set_id=payload_res.get("payload_dataset_id")
    )
    logger.info(
        f"subscriptions {subscription_id} created for deployment {deployment_name}"
    )
    return subscription_id
