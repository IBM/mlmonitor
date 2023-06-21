# SPDX-License-Identifier: Apache-2.0
import os
import random
import uuid
import pandas as pd
import requests

from ibm_watson_openscale.supporting_classes.enums import DataSetTypes, TargetTypes
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import (
    Target,
    CustomCredentials,
)
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.service_provider import delete_provider, add_custom_provider
from mlmonitor.src.wos.data_mart import create_datamart
from mlmonitor.src import DATA_ROOT, IAM_URL, logger
from mlmonitor.data import BUCKET_NAME, COS_ENDPOINT, COS_RESOURCE_CRN, CLOUD_API_KEY
from mlmonitor.data.cos import put_item
from mlmonitor.src.wos.subscription import (
    create_binary_classification_subscription_custom,
)

from mlmonitor.src.model.config import ModelConfig


def custom_ml_scoring(
    scoring_url: str, request_headers: dict, df: pd.DataFrame, format: str = "wos"
) -> dict:
    if format == "aws":
        payload = {
            "instances": [{"features": features} for features in df.values.tolist()]
        }
    elif format == "wos":
        payload = {"fields": df.columns.tolist(), "values": df.values.tolist()}
    else:
        raise ValueError("format should be aws or wos")

    scoring_response = requests.post(
        scoring_url,
        json=payload,
        headers=request_headers,
        verify=False,
    )

    return scoring_response.json()


def get_wos_response_custom(
    scoring_url: str,
    df: pd.DataFrame,
    request_headers: dict,
    format: str = "wos",
    prediction_field: str = "prediction",
    probability_field: str = "probability",
):
    resp = custom_ml_scoring(
        scoring_url=scoring_url, df=df, request_headers=request_headers, format=format
    )

    if format == "aws":
        return {
            "fields": [prediction_field, probability_field],
            "values": [list(x.values()) for x in resp.get("predictions")],
        }

    elif format == "wos":
        return resp
    else:
        raise ValueError("format should be aws or wos")


# https://www.ibm.com/docs/en/cloud-paks/cp-data/3.0.1?topic=models-custom-ml-frameworks
def monitor_custom_model(
    model_config: ModelConfig,
    scoring_url: str,
    deployment_name: str,
    credentials: CustomCredentials,
    request_headers: dict,
    wos_provider_type: str,
    wos_provider_name: str,
    wos_provider_description: str,
    wos_provider_keep: bool = True,
    inference_samples: int = 2,
    data_path: str = DATA_ROOT,
) -> dict:
    """Configure a Watson OpenScale subscription for an online Endpoint
    As a result of this action , the selected model should be Monitored for Quality and  explainability with <wos_provider_name>
    as Machine Learning provider.
    ----------
    source_dir : str
       this indicates the location of source code and dependencies to be uploaded and used by the endpoint
    scoring_url : str
       url of the custom ml provider
    deployment_name : str
       this indicates the Endpoint Name in Sagemaker for which an OpenScale subscription should be created
    request_headers : dict
       headers to use by the custom provider:
    credentials : dict
       credentials to use by the custom provider:
    wos_provider_type : str
       type of Watson OpenScale ML service provider to create if it does not already exists. options are
       production or pre_production
    wos_provider_name : str
       this indicates OpenScale ML service provider name
       to use (if exists and wos_provider_keep=True)
       or create (if does not exists and wos_provider_keep=False)
    wos_provider_description : str
       this indicates description to add to OpenScale ML service provider if created (does not exists yet and wos_provider_keep=False)
    wos_provider_keep : bool
       this boolean flag indicates whether an existing OpenScale ML service provider with the same name should be reused or re-created
    inference_samples : int
       Number of inference samples to send in the scoring request to the deployed Endpoint (deployment_name)
    data_path : str
       location of dataset to be fetched to get scoring request samples
    Returns
    -------
    deployment_info : dict
        dictionary with model details and Watson OpenScale subscription infos
        {
        "features": [],
        "target": "<model target column>"
        "job_name": "<sagemaker training job name>",
        "model_data": "<model.tar.gz location>",
        "model_endpoint": "<sagemaker endpoint name if already>",
        "data_mart_id": "<Watson OpenScale Data Mart ID  if model already monitored>",
        "service_provider_id": "3182bc7d-403a-4d24-82e7-f2d150714de2",
        "subscription_id": "<Watson OpenScale subscription ID if model already monitored>",
        }
    """

    training_data_file_name = model_config.training_data
    categorical_fields = model_config.categorical_columns
    prediction_field = model_config.prediction_field
    probability_fields = model_config.probability_fields
    scoring_df = pd.read_csv(
        os.path.join(data_path, training_data_file_name), engine="python"
    )
    put_item(item_name=training_data_file_name, item_path=data_path)

    features = model_config.feature_columns
    target = model_config.class_label

    logger.debug(
        f"Deployment Name {deployment_name}\n"
        f"Provider Name [{wos_provider_name}] to create\n"
        f"WOS provider type [{wos_provider_type}]"
        f"Number features [{len(features)}]"
        f"target [{target}]"
    )

    if not wos_provider_keep:
        delete_provider(wos_client=wos_client, service_provider_name=wos_provider_name)

    # ! TODO Custom provider specific
    #  https://github.com/IBM/monitor-custom-ml-engine-with-watson-openscale/blob/master/notebooks/WatsonOpenScaleAndCustomMLEngine.ipynb

    service_provider_id = add_custom_provider(
        wos_client=wos_client,
        wos_provider_type=wos_provider_type,
        service_provider_name=wos_provider_name,
        service_provider_descr=wos_provider_description,
        ml_credentials=credentials,
        request_headers={"Content-Type": "application/json"},
    )

    logger.info(
        f"Service provider {wos_provider_name} created with Provider id {service_provider_id}"
    )

    data_mart_id = create_datamart(wos_client=wos_client)

    subscription_name = "ocp custom provider"
    asset_name = f"[asset] {subscription_name}"
    subscriptions = wos_client.subscriptions.list().result.subscriptions
    for subscription in subscriptions:
        print(f"subscription.entity.asset.name {subscription.entity.asset.name}")
        if subscription.entity.asset.name == asset_name:
            sub_model_id = subscription.metadata.id
            wos_client.subscriptions.delete(subscription.metadata.id)
            print("Deleted existing subscription for model", sub_model_id)

    # ! TODO Custom provider specific
    if model_config.problem_type == "binary":
        # scoring_request_headers = {"Content-Type": "application/json"}
        scoring_request_headers = {}
        subscription_id = create_binary_classification_subscription_custom(
            wos_client=wos_client,
            service_provider_id=service_provider_id,
            data_mart_id=data_mart_id,
            scoring_url=scoring_url,
            deployment_name=deployment_name,
            asset_name=asset_name,
            scoring_request_headers=scoring_request_headers,
            bucket_name=BUCKET_NAME,
            cos_resource_crn=COS_RESOURCE_CRN,
            cos_endpoint=COS_ENDPOINT,
            apikey=CLOUD_API_KEY,
            iam_url=IAM_URL,
            features=features,
            categorical_fields=categorical_fields,
            target=target,
            prediction_field=prediction_field,
            probability_fields=probability_fields,
            training_data_file_name=training_data_file_name,
        )
    else:
        raise ValueError(f"{model_config.problem_type} problem type not implemented")

    logger.info(
        f"subscriptions {subscription_id} created for deployment {deployment_name}"
    )

    payload_data_set_id = (
        wos_client.data_sets.list(
            type=DataSetTypes.PAYLOAD_LOGGING,
            target_target_id=subscription_id,
            target_target_type=TargetTypes.SUBSCRIPTION,
        )
        .result.data_sets[0]
        .metadata.id
    )
    if payload_data_set_id is None:
        raise RuntimeError(
            "Payload dataset not found. Please check subscription status."
        )
    else:
        logger.info(f"Payload data set id: {payload_data_set_id}")

    mask = random.sample(range(1, scoring_df.shape[0] + 1), inference_samples)
    scoring_df = scoring_df.loc[:, features].iloc[mask, :]

    # scoring_payload_wos = {"fields": features, "values": scoring_df.iloc[mask, 1:].values.tolist()}
    scoring_payload_wos = {
        "fields": features,
        "values": [
            list(x.values()) for x in scoring_df.to_dict(orient="index").values()
        ],
    }

    # ! TODO Custom provider specific
    scoring_response_wos = get_wos_response_custom(
        df=scoring_df,
        scoring_url=scoring_url,
        request_headers=request_headers,
        probability_field=model_config.probability_fields[0],
        prediction_field=model_config.prediction_field,
    )

    logger.debug(f"scoring_payload_wos : {scoring_payload_wos}")
    logger.debug(f"scoring_response_wos : {scoring_response_wos}")

    logger.info(f"explicit payload logging with {scoring_df.shape[0]} requests .")
    wos_client.data_sets.store_records(
        data_set_id=payload_data_set_id,
        background_mode=False,
        request_body=[
            PayloadRecord(
                scoring_id=str(uuid.uuid4()),
                request=scoring_payload_wos,
                response=scoring_response_wos,
                response_time=460,
            )
        ],
    )

    wos_client.data_sets.show_records(payload_data_set_id, limit=1)
    pl_records_count = wos_client.data_sets.get_records_count(payload_data_set_id)
    logger.debug(f"Number of records in the payload logging table: {pl_records_count}")

    # Enable MRM Monitor
    mrm_monitor_details = wos_client.monitor_instances.create(
        data_mart_id=data_mart_id,
        background_mode=False,
        monitor_definition_id=wos_client.monitor_definitions.MONITORS.MODEL_RISK_MANAGEMENT_MONITORING.ID,
        target=Target(target_type=TargetTypes.SUBSCRIPTION, target_id=subscription_id),
        parameters={},
    ).result

    mrm_instance_id = mrm_monitor_details.metadata.id
    logger.info(f"MRM Monitor ID Created [{mrm_instance_id}]")

    quality_monitor_details = wos_client.monitor_instances.create(
        data_mart_id=data_mart_id,
        background_mode=False,
        monitor_definition_id=wos_client.monitor_definitions.MONITORS.QUALITY.ID,
        target=Target(target_type=TargetTypes.SUBSCRIPTION, target_id=subscription_id),
        parameters=model_config.quality_monitor_parameters,
        thresholds=model_config.quality_monitor_thresholds,
    ).result

    quality_monitor_instance_id = quality_monitor_details.metadata.id
    logger.info(f"Quality Monitor ID Created [{quality_monitor_instance_id}]")

    # Configure EXPLAINABILITY
    parameters = {"enabled": True}
    explainability_details = wos_client.monitor_instances.create(
        data_mart_id=data_mart_id,
        background_mode=False,
        monitor_definition_id=wos_client.monitor_definitions.MONITORS.EXPLAINABILITY.ID,
        target=Target(target_type=TargetTypes.SUBSCRIPTION, target_id=subscription_id),
        parameters=parameters,
    ).result
    explainability_monitor_id = explainability_details.metadata.id
    logger.info(f"Explainability Monitor ID Created [{explainability_monitor_id}]")

    wos_client.data_sets.print_records_schema(data_set_id=payload_data_set_id)

    deployment_info = {
        "service_provider_id": service_provider_id,
        "subscription_id": subscription_id,
        "data_mart_id": data_mart_id,
        "features": features,
        "target": target,
    }

    return deployment_info
