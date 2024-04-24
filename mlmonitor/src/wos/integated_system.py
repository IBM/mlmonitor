# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale import APIClient
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import MonitorMetricRequest
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import MetricThreshold
from ibm_watson_openscale.supporting_classes.enums import MetricThresholdTypes
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import MonitorTagRequest
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import IntegratedSystems


def create_integrated_system_credentials_cp4d(username: str, url: str, api_key: str):
    return {
        "auth_type": "bearer",
        "token_info": {
            "url": f"{url}/icp4d-api/v1/authorize",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            "payload": {"username": username, "api_key": api_key},
            "method": "post",
        },
    }


def create_integrated_system_credentials_cloud(url: str, api_key: str):
    return {
        "auth_type": "bearer",
        "token_info": {
            "url": url,  # update the token generation here
            "headers": {
                "Content-type": "application/x-www-form-urlencoded"
            },  # update the headers here
            "payload": f"grant_type=urn:ibm:params:oauth:grant-type:apikey&response_type=cloud_iam&apikey={api_key}",
            # update the payload here
            "method": "POST",  # update the http method here
        },
    }


def create_integrated_system(
    wos_client: APIClient,
    provider_name: str,
    auth_url: str,
    api_key: str,
    scoring_url: str,
    username: str = None,
):
    if username:
        custom_provider_creds = create_integrated_system_credentials_cp4d(
            username=username, url=auth_url, api_key=api_key
        )
    else:
        custom_provider_creds = create_integrated_system_credentials_cloud(
            auth_url, api_key
        )

    custom_metrics_integrated_system = (
        IntegratedSystems(wos_client)
        .add(
            name=provider_name,
            description=provider_name,
            type="custom_metrics_provider",
            credentials=custom_provider_creds,
            connection={"display_name": provider_name, "endpoint": scoring_url},
        )
        .result
    )

    return custom_metrics_integrated_system


def get_integrated_system_by_provider_name(wos_client: APIClient, provider_name: str):
    return [
        system.metadata.id
        for system in IntegratedSystems(wos_client).list().result.integrated_systems
        if system.entity.name == provider_name
        and system.entity.type == "custom_metrics_provider"
    ]


def get_custom_monitor_by_name(wos_client: APIClient, provider_name: str):
    return [
        system.metadata.id
        for system in IntegratedSystems(wos_client).list().result.integrated_systems
        if system.entity.name == provider_name
        and system.entity.type == "custom_metrics_provider"
    ]


def create_custom_metric_definitions(
    custom_metrics_names: tuple, custom_metrics_thresholds: tuple
):
    # Update the tag values if you want to fetch the metrics by tags
    TAGS = ["region"]
    TAG_DESCRIPTION = ["customer geographical region"]

    metrics = [
        MonitorMetricRequest(
            name=name,
            thresholds=[
                MetricThreshold(
                    type=MetricThresholdTypes.LOWER_LIMIT, default=threshold
                )
            ],
        )
        for name, threshold in zip(custom_metrics_names, custom_metrics_thresholds)
    ]

    # Comment the below tags code if there are no tags to be created
    tags = [MonitorTagRequest(name=TAGS[0], description=TAG_DESCRIPTION[0])]

    return metrics, tags
