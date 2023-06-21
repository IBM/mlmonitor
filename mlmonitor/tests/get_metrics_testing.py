# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale import APIClient as WOS_APIClient
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json

from custmonitor.metricsprovider.helpers import get_access_token_cloud

from mlmonitor.src import get_connection_details, logger

from mlmonitor.src.utils.utils import parse_args
from mlmonitor.src.wml import wml_client, WML_SPACE_ID
from mlmonitor.src import API_KEY, WOS_URL
from mlmonitor.src.wml.utils import get_deployment_uid_by_name
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.wos.data_mart import get_datamart_ids
from mlmonitor.src.wos.custom_monitors import (
    get_custom_monitor_definition,
    get_custom_monitor_instance,
)
from mlmonitor.src.wos.integated_system import get_integrated_system_by_provider_name


def test_get_metrics_local(monitored_deployment: str, source_dir: str):
    API_KEY, AUTH_ENDPOINT = get_connection_details()
    wos_client = WOS_APIClient(authenticator=IAMAuthenticator(apikey=API_KEY))
    token = get_access_token_cloud(apikey=API_KEY)

    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=monitored_deployment
    )
    data_marts = get_datamart_ids(wos_client=wos_client)
    data_mart_id = data_marts[0]
    if len(subscription_ids) == 1:
        subscription_id = subscription_ids[0]
    else:
        raise ValueError(
            f"No WOS subscription found for deployment {monitored_deployment}"
        )

    if source_dir == "use_case_churn":
        from custmonitor.metrics.customer_churn import get_metrics
    elif source_dir == "use_case_gcr":
        from custmonitor.metrics.credit_risk import get_metrics
    else:
        raise ValueError(
            f"Unsupported model use case {source_dir} should be in ['use_case_churn','use_case_gcr']"
        )

    res = get_metrics(token, data_mart_id, subscription_id, WOS_URL)

    print(res)
    return res


def test_get_metrics_wml(
    monitored_deployment: str,
    provider_name: str,
    custom_monitor_name: str,
    wml_function_deployment_name: str,
    test: True,
):
    """
    The test_get_metrics_wml function is a test function that calls the get_metrics_wml function
    with some sample data. It returns the extracted metrics from the WML deployment with id
    [deployment_uid] and prints them to stdout.

    :param monitored_deployment:str: Get the deployment uid of the model endpoint for which
    :param provider_name:str: Retrieve the integrated system id from wml
    :param custom_monitor_name:str: Select the custom monitor to be used
    :param wml_function_deployment_name:str: Get the deployment_uid:str parameter
    :param test:True: Obtain the values of the metrics returned by wml
    :return: A tuple of the form:

    """
    wos_client = WOS_APIClient(authenticator=IAMAuthenticator(apikey=API_KEY))
    wml_client.set.default_space(WML_SPACE_ID)

    deployment_uid = get_deployment_uid_by_name(
        wml_client=wml_client, deployment_name=wml_function_deployment_name
    )

    # Deployment (Model endpoint) for which the Custom monitor is attached (via subscription)
    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=monitored_deployment
    )
    data_marts = get_datamart_ids(wos_client=wos_client)
    data_mart_id = data_marts[0]
    if len(subscription_ids) == 1:
        subscription_id = subscription_ids[0]
    else:
        raise ValueError(
            f"No WOS subscription found for deployment {monitored_deployment}"
        )

    if test:
        input_data = {
            "input_data": [
                {
                    "values": {
                        "data_mart_id": data_mart_id,
                        "subscription_id": subscription_id,
                        "test": "test",
                        "custom_monitor_run_id": "123",
                        "custom_monitor_id": "not needed",
                        "custom_monitor_instance_id": "not needed",
                        "custom_monitor_instance_params": {
                            "custom_metrics_provider_id": "not needed",
                            "custom_metrics_wait_time": 300,
                        },
                    }
                }
            ]
        }

        scoring_response_test = wml_client.deployments.score(deployment_uid, input_data)
        print(
            "response_payload for custom metrics provider test mode\n",
            json.dumps(scoring_response_test, indent=3),
        )
        return scoring_response_test

    custom_metrics_provider_id = get_integrated_system_by_provider_name(
        wos_client, provider_name
    )

    if len(custom_metrics_provider_id) == 1:
        integrated_system_id = custom_metrics_provider_id[0]
    else:
        raise ValueError(f"custom provider not found with name : [{provider_name}]")

    if not (
        existing_definition := get_custom_monitor_definition(
            wos_client=wos_client, monitor_name=custom_monitor_name
        )
    ):
        raise ValueError(
            f"custom monitor definition not found for monitor : [{custom_monitor_name}]"
        )

    monitor_definition_id = existing_definition.metadata.id
    existing_monitor_instance = get_custom_monitor_instance(
        wos_client=wos_client,
        data_mart_id=data_mart_id,
        monitor_definition_id=monitor_definition_id,
        subscription_id=subscription_id,
    )
    monitor_instance_id = existing_monitor_instance.metadata.id
    if not existing_monitor_instance:
        raise ValueError(
            f"custom monitor instance not found for monitor : [{custom_monitor_name}]"
        )
    print(
        f"- Found integrated service id [{integrated_system_id}] => [{existing_monitor_instance.entity.parameters.get('custom_metrics_provider_id')}]\n"
        f"- Found custom monitor definition id [{monitor_definition_id}] => [{existing_monitor_instance.entity.monitor_definition_id}]\n"
        f"- Found custom monitor instance id [{monitor_instance_id}]"
    )

    input_data = {
        "input_data": [
            {
                "values": {
                    "data_mart_id": data_mart_id,
                    "subscription_id": subscription_id,
                    "custom_monitor_run_id": "123",
                    "custom_monitor_id": monitor_definition_id,
                    "custom_monitor_instance_id": monitor_instance_id,
                    "custom_monitor_instance_params": {
                        "custom_metrics_provider_id": custom_metrics_provider_id,
                        "custom_metrics_wait_time": 300,
                    },
                }
            }
        ]
    }
    scoring_response = wml_client.deployments.score(deployment_uid, input_data)
    print(
        "response_payload for custom metrics provider wml function\n",
        json.dumps(scoring_response, indent=3),
    )
    return scoring_response


if __name__ == "__main__":
    import importlib

    args = parse_args()
    # Model for which the Custom monitor is attached (via subscription)

    model_signature = importlib.import_module(
        f"mlmonitor.{args.source_dir}.model_signature"
    )
    custom_monitor = getattr(model_signature, "custom_monitor")
    wml_function_provider = custom_monitor.get("wml_function_provider")
    logger.info(f"get metrics for Custom WML function {wml_function_provider}-deploy")

    monitored_deployment = args.deployment_name
    source_dir = args.source_dir

    provider_name = custom_monitor.get("provider_name")
    custom_monitor_name = custom_monitor.get("custom_monitor_name")
    wml_function_deployment_name = f"{wml_function_provider}-deploy"

    test_get_metrics_local(
        monitored_deployment=monitored_deployment, source_dir=source_dir
    )

    test_get_metrics_wml(
        monitored_deployment=monitored_deployment,
        provider_name=provider_name,
        custom_monitor_name=custom_monitor_name,
        wml_function_deployment_name=wml_function_deployment_name,
        test=True,
    )
