# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale import APIClient
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import (
    SageMakerCredentials,
    CustomCredentials,
    WMLCredentialsCloud,
    AzureCredentials,
)
from ibm_watson_openscale.supporting_classes.enums import ServiceTypes


def get_provider_id(wos_client: APIClient, service_provider_name: str):
    filtered_service_providers = [
        sp.metadata.id
        for sp in wos_client.service_providers.list().result.service_providers
        if sp.entity.name == service_provider_name
    ]
    if len(filtered_service_providers) == 1:
        return filtered_service_providers[0]
    elif len(filtered_service_providers) > 1:
        raise ValueError(
            f"{len(filtered_service_providers)} providers found , at most 1 expected"
        )
    else:
        return None


def delete_provider(wos_client: APIClient, service_provider_name: str):
    if service_provider_id := get_provider_id(
        wos_client=wos_client, service_provider_name=service_provider_name
    ):
        wos_client.service_providers.delete(service_provider_id, background_mode=False)
        print(f"Deleted existing service_provider : {service_provider_id}")
    else:
        print(f"No provider ID found for name : {service_provider_name}")


def add_aws_provider(
    wos_client: APIClient,
    service_provider_name: str,
    service_provider_descr: str,
    access_key_id: str,
    secret_access_key: str,
    region: str,
    wos_provider_type: str = "pre_production",
):
    service_provider_id = get_provider_id(
        wos_client=wos_client, service_provider_name=service_provider_name
    )

    if not service_provider_id:
        added_service_provider_result = wos_client.service_providers.add(
            name=service_provider_name,
            description=service_provider_descr,
            service_type=ServiceTypes.AMAZON_SAGEMAKER,
            operational_space_id=wos_provider_type,
            credentials=SageMakerCredentials(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                region=region,
            ),
            background_mode=False,
        ).result
        return added_service_provider_result.metadata.id
    else:
        print(
            f"AWS provider already found for name : {service_provider_name} with id {service_provider_id}"
        )
        return service_provider_id


def add_custom_provider(
    wos_client: APIClient,
    service_provider_name: str,
    service_provider_descr: str,
    ml_credentials: CustomCredentials,
    request_headers: dict,
    wos_provider_type: str = "pre_production",
):
    if service_provider_id := get_provider_id(
        wos_client=wos_client, service_provider_name=service_provider_name
    ):
        print(
            f"Custom provider already found for name : {service_provider_name} with id {service_provider_id}"
        )
        return service_provider_id
    else:
        added_service_provider_result = wos_client.service_providers.add(
            name=service_provider_name,
            description=service_provider_descr,
            service_type=ServiceTypes.CUSTOM_MACHINE_LEARNING,
            request_headers=request_headers,
            operational_space_id=wos_provider_type,
            credentials=ml_credentials,
            background_mode=False,
        ).result
        return added_service_provider_result.metadata.id


def add_wml_provider(
    wos_client: APIClient,
    service_provider_name: str,
    service_provider_descr: str,
    space_id: str,
    apikey: str,
    url: str = "https://us-south.ml.cloud.ibm.com",
    wos_provider_type: str = "pre_production",
):
    service_provider_id = get_provider_id(
        wos_client=wos_client, service_provider_name=service_provider_name
    )

    if not service_provider_id:
        added_service_provider_result = wos_client.service_providers.add(
            name=service_provider_name,
            description=service_provider_descr,
            service_type=ServiceTypes.WATSON_MACHINE_LEARNING,
            deployment_space_id=space_id,
            operational_space_id=wos_provider_type,
            credentials=WMLCredentialsCloud(apikey=apikey, url=url, instance_id=None),
            background_mode=False,
        ).result
        return added_service_provider_result.metadata.id
    else:
        print(
            f"WML provider already found for name : {service_provider_name} with id {service_provider_id}"
        )
        return service_provider_id


def add_azure_provider(
    wos_client: APIClient,
    service_provider_name: str,
    service_provider_descr: str,
    subscription_id: str,
    client_secret: str,
    client_id: str,
    tenant_id: str,
    wos_provider_type: str = "pre_production",
):
    service_provider_id = get_provider_id(
        wos_client=wos_client, service_provider_name=service_provider_name
    )
    if not service_provider_id:
        service_type = "azure_machine_learning_service"
        added_service_provider_result = wos_client.service_providers.add(
            name=service_provider_name,
            description=service_provider_descr,
            service_type=service_type,
            operational_space_id=wos_provider_type,
            credentials=AzureCredentials(
                subscription_id=subscription_id,
                client_id=client_id,
                client_secret=client_secret,
                tenant=tenant_id,
            ),
            background_mode=False,
        ).result
        return added_service_provider_result.metadata.id
    else:
        print(
            f"Azure provider already found for name : {service_provider_name} with id {service_provider_id}"
        )
        return service_provider_id
