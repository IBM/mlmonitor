# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale import APIClient
import uuid
from ibm_watson_openscale.supporting_classes.enums import (
    AssetTypes,
    InputDataType,
    ProblemType,
    DeploymentTypes,
)
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import (
    Asset,
    AssetDeploymentRequest,
    ScoringEndpointRequest,
    AssetPropertiesRequest,
    TrainingDataReference,
    COSTrainingDataReferenceLocation,
    COSTrainingDataReferenceConnection,
)


def get_subscription_id_by_deployment(wos_client: APIClient, deployment_name: str):
    return [
        sub.metadata.id
        for sub in wos_client.subscriptions.list().result.subscriptions
        if sub.entity.deployment.name == deployment_name
    ]


def get_subscription_id_by_model(wos_client: APIClient, model_name: str):
    return [
        sub.metadata.id
        for sub in wos_client.subscriptions.list().result.subscriptions
        if sub.entity.asset.name == model_name
    ]


def get_subscription_details_by_deployment(wos_client: APIClient, deployment_name: str):
    return [
        sub.to_dict()
        for sub in wos_client.subscriptions.list().result.subscriptions
        if sub.entity.deployment.name == deployment_name
    ]


def delete_subscription(wos_client: APIClient, deployment_name: str):
    count = 0
    subscription_ids = get_subscription_id_by_deployment(wos_client, deployment_name)
    for subscription_id in subscription_ids:
        wos_client.subscriptions.delete(subscription_id)
        count += 1
    return count


def create_classification_subscription(
    wos_client: APIClient,
    service_provider_id: str,
    data_mart_id: str,
    model_asset_details: dict,
    bucket_name: str,
    cos_resource_crn: str,
    cos_endpoint: str,
    apikey: str,
    iam_url: str,
    features: list,
    target: str,
    training_data_file_name: str,
    probability_fields: list,
    prediction_field: str,
    categorical_fields: list,
    classification_type: str,
    data_type: str,
    env: str = "saas",
):
    if classification_type == "binary":
        problem_type = ProblemType.BINARY_CLASSIFICATION
    elif classification_type == "multiclass":
        problem_type = ProblemType.MULTICLASS_CLASSIFICATION
    else:
        raise ValueError("classification_type should be binary or multiclass ")

    if training_data_file_name:
        # !TODO only support COS for CP4D SaaS version , need support DB2 for CP4D on prem
        training_data_reference = TrainingDataReference(
            type="cos",
            location=COSTrainingDataReferenceLocation(
                bucket=bucket_name, file_name=training_data_file_name
            ),
            connection=COSTrainingDataReferenceConnection.from_dict(
                {
                    "resource_instance_id": cos_resource_crn,
                    "url": cos_endpoint,
                    "api_key": apikey,
                    "iam_url": iam_url,
                }
            ),
        )

    else:

        training_data_reference = None

    if data_type == "structured":
        input_data_type = InputDataType.STRUCTURED
    elif data_type == "unstructured_image":
        input_data_type = InputDataType.UNSTRUCTURED_IMAGE
    else:
        raise ValueError("data_type should be structured or unstructured_image ")

    # TODO compare 2 urls
    url = (
        model_asset_details["metadata"]["url"]
        if env == "saas"
        else model_asset_details["entity"]["scoring_endpoint"]["url"]
    )

    subscription_details = wos_client.subscriptions.add(
        background_mode=False,
        data_mart_id=data_mart_id,
        service_provider_id=service_provider_id,
        asset=Asset(
            asset_id=model_asset_details["entity"]["asset"]["asset_id"],
            name=model_asset_details["entity"]["asset"]["name"],
            url=model_asset_details["entity"]["asset"]["url"],
            asset_type=AssetTypes.MODEL,
            input_data_type=input_data_type,
            problem_type=problem_type,
        ),
        deployment=AssetDeploymentRequest(
            deployment_id=model_asset_details["metadata"]["guid"],
            name=model_asset_details["entity"]["name"],
            deployment_type=DeploymentTypes.ONLINE,
            url=url,
            scoring_endpoint=ScoringEndpointRequest(
                url=model_asset_details["entity"]["scoring_endpoint"]["url"]
            ),
        ),
        asset_properties=AssetPropertiesRequest(
            label_column=target,
            probability_fields=probability_fields,
            prediction_field=prediction_field,
            feature_fields=features,
            categorical_fields=categorical_fields,
            training_data_reference=training_data_reference,
        ),
    ).result

    return subscription_details.metadata.id


def create_binary_classification_subscription_custom(
    wos_client: APIClient,
    scoring_url: str,
    scoring_request_headers: dict,
    deployment_name: str,
    asset_name: str,
    service_provider_id: str,
    data_mart_id: str,
    bucket_name: str,
    cos_resource_crn: str,
    cos_endpoint: str,
    apikey: str,
    iam_url: str,
    features: list,
    target: str,
    training_data_file_name: str,
    probability_fields: list,
    prediction_field: str,
    categorical_fields: list,
):
    asset_id = str(uuid.uuid4())
    url = ""

    asset_deployment_id = str(uuid.uuid4())

    subscription_details = wos_client.subscriptions.add(
        background_mode=False,
        data_mart_id=data_mart_id,
        service_provider_id=service_provider_id,
        asset=Asset(
            asset_id=asset_id,
            name=asset_name,
            url=url,
            asset_type=AssetTypes.MODEL,
            input_data_type=InputDataType.STRUCTURED,
            problem_type=ProblemType.BINARY_CLASSIFICATION,
        ),
        deployment=AssetDeploymentRequest(
            deployment_id=asset_deployment_id,
            name=deployment_name,
            deployment_type=DeploymentTypes.ONLINE,
            scoring_endpoint=ScoringEndpointRequest(
                url=scoring_url,
                request_headers=scoring_request_headers,
                credentials=None,
            ),
        ),
        asset_properties=AssetPropertiesRequest(
            label_column=target,
            probability_fields=probability_fields,
            prediction_field=prediction_field,
            feature_fields=features,
            categorical_fields=categorical_fields,
            training_data_reference=TrainingDataReference(
                type="cos",
                location=COSTrainingDataReferenceLocation(
                    bucket=bucket_name, file_name=training_data_file_name
                ),
                connection=COSTrainingDataReferenceConnection.from_dict(
                    {
                        "resource_instance_id": cos_resource_crn,
                        "url": cos_endpoint,
                        "api_key": apikey,
                        "iam_url": iam_url,
                    }
                ),
            ),
        ),
    ).result

    return subscription_details.metadata.id
