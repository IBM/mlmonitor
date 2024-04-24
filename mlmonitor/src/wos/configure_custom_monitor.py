# SPDX-License-Identifier: Apache-2.0
import json
import importlib
import ibm_watson_openscale
import ibm_watson_machine_learning

from mlmonitor.src.wos.data_mart import get_datamart_ids
from mlmonitor.src import API_KEY, IAM_URL, logger
from mlmonitor.src.utils.utils import parse_args
from mlmonitor.src.wml import wml_client, WML_SPACE_ID
from mlmonitor.src.wml.scoring import get_deployment_scoring_url
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.wos.custom_monitors import (
    get_integrated_system_by_provider_name,
    create_custom_monitor_definition,
    get_custom_monitor_instance,
    create_custom_monitor_instance,
)
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.integated_system import create_integrated_system


def configure_custom_monitor(
    wos_client: ibm_watson_openscale.APIClient,
    wml_client: ibm_watson_machine_learning.APIClient,
    monitored_model: str,
    deployment_name: str,
    provider_name: str,
    custom_monitor_name: str,
    custom_metrics_names: tuple,
    custom_metrics_thresholds: tuple,
    wml_space_id: str = WML_SPACE_ID,
    apikey: str = API_KEY,
    auth_url: str = IAM_URL,
    username: str = None,
) -> dict:
    """
    The configure_custom_monitor function :
        - Creates a custom monitor definition for the given model.
        - Creates a custom monitor instance for the given model.
        - Runs the custom monitor instance to evaluate and capture metrics from your deployed model.

    :param wos_client:ibm_watson_openscale.APIClient: Watson OpenScale Client
    :param wml_client:ibm_watson_machine_learning.APIClient: Watson Machine Learning Client
    :param monitored_model:str: Specify the model deployment name monitored in Watson Openscale
    :param deployment_name:str: Specify the name of WML Custom Provider deployment function already deployed
    :param provider_name:str: unique name for the integrated system
    :param custom_monitor_name:str: custom monitor definition
    :param custom_metrics_names:tuple: Pass the names of the custom metrics that you want to be monitored
    :param custom_metrics_thresholds:tuple: Configure thresholds for each custom metric
    :param wml_space_id:str=WML_SPACE_ID: Watson Machine Learning Deployment Space ID where WML Custom Provider function is deployed
    :param apikey:str=API_KEY: Watson OpenScale Client API Key
    :param auth_url:str=IAM_URL: Specify the url of your IAM instance
    :param username:str=None: CP4D username
    :return: A dict with the following structure:

    """

    logger.info(
        f"Create a Custom Monitor for with:\n"
        f"Custom Metrics provider name [{provider_name}] to create\n"
        f"wml_function_provider {deployment_name}\n"
        f"Custom Monitor name [{custom_monitor_name}] to create\n"
        f"Custom metrics name [{custom_metrics_names}]\n"
        f"Custom metrics thresholds {custom_metrics_thresholds}"
    )

    wml_client.set.default_space(wml_space_id)

    data_marts = get_datamart_ids(wos_client=wos_client)
    scoring_url = get_deployment_scoring_url(
        wml_client=wml_client, deployment_name=deployment_name
    )
    scoring_url = f"{scoring_url}?version=2023-03-31"
    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=monitored_model
    )

    if len(subscription_ids) == 1:
        subscription_id = subscription_ids[0]
    else:
        raise ValueError(
            f"You should Create a WOS subscription for deployment {monitored_model} first"
        )
    data_mart_id = data_marts[0]

    integrated_system_ids = get_integrated_system_by_provider_name(
        wos_client=wos_client, provider_name=provider_name
    )

    if len(integrated_system_ids) == 1:
        logger.info(f"Integrated system ID {integrated_system_ids[0]}")
        integrated_system_id = integrated_system_ids[0]

    elif len(integrated_system_ids) == 0:
        custom_metrics_integrated_system = create_integrated_system(
            wos_client=wos_client,
            provider_name=provider_name,
            api_key=apikey,
            scoring_url=scoring_url,
            auth_url=auth_url,
            username=username,
        )
        integrated_system_id = custom_metrics_integrated_system.metadata.id
        logger.info(
            f"IntegratedSystems created with name {provider_name} with id {integrated_system_id}"
        )

    else:
        raise ValueError("integrated_system_ids")

    logger.info(
        "WML function deployed for this custom monitor [{deployment_name}]\n"
        f"Subscription ID [{subscription_id}]\n"
        f"Provider Name [{provider_name}] to create\n"
        f"Scoring url for WML function [{scoring_url}]\n"
        f"Monitored model Name [{monitored_model}]\n"
        f"Data Mart ID [{data_mart_id}]\n"
        f"custom_monitor_name [{custom_monitor_name}]"
    )

    custom_monitor_details = create_custom_monitor_definition(
        wos_client=wos_client,
        custom_monitor_name=custom_monitor_name,
        custom_metrics_names=custom_metrics_names,
        custom_metrics_thresholds=custom_metrics_thresholds,
        enable_schedule=True,
    )

    monitor_definition_id = custom_monitor_details.metadata.id

    logger.info(f"Custom metrics definition id {monitor_definition_id}")

    existing_monitor_instance = get_custom_monitor_instance(
        wos_client=wos_client,
        data_mart_id=data_mart_id,
        monitor_definition_id=monitor_definition_id,
        subscription_id=subscription_id,
    )

    logger.info(f"Existing custom monitor instances {existing_monitor_instance}")
    logger.debug(monitor_definition_id)

    custom_monitor_instance_details = create_custom_monitor_instance(
        wos_client=wos_client,
        data_mart_id=data_mart_id,
        monitor_definition_id=monitor_definition_id,
        subscription_id=subscription_id,
        integrated_system_id=integrated_system_id,
        custom_metrics_names=custom_metrics_names,
        custom_metrics_thresholds=custom_metrics_thresholds,
        custom_metrics_wait_time=300,
        max_records=None,
        min_records=None,
    )

    custom_monitor_instance_id = custom_monitor_instance_details.metadata.id

    # Run Custom Monitor after creation
    monitor_instance_run_info = wos_client.monitor_instances.run(
        background_mode=False, monitor_instance_id=custom_monitor_instance_id
    ).result

    logger.info(
        f"Custom Monitor Run Evaluation result :\n{json.dumps(monitor_instance_run_info.to_dict(), indent=4)}"
    )

    return custom_monitor_instance_details


if __name__ == "__main__":
    args = parse_args()

    model_signature = importlib.import_module(
        f"mlmonitor.{args.source_dir}.model_signature"
    )
    custom_monitor = getattr(model_signature, "custom_monitor")
    wml_function_provider = custom_monitor.get("wml_function_provider")

    # New Custom Monitor instance will be tied to the WOS subscription of this model
    monitored_model = args.deployment_name
    # Deployment name corresponds to the WML function deployed for this custom monitor
    deployment_name = f"{wml_function_provider}-deploy"

    # CUSTOM MONITOR SPECIFIC NAMES
    provider_name = custom_monitor.get("provider_name")
    # Name Displayed in WOS UI
    custom_monitor_name = custom_monitor.get("custom_monitor_name")
    custom_metrics_names = tuple(custom_monitor.get("names"))
    custom_metrics_thresholds = tuple(custom_monitor.get("thresholds"))

    custom_monitor_instance_details = configure_custom_monitor(
        monitored_model=monitored_model,
        wos_client=wos_client,
        wml_client=wml_client,
        deployment_name=deployment_name,
        provider_name=provider_name,
        custom_monitor_name=custom_monitor_name,
        custom_metrics_names=custom_metrics_names,
        custom_metrics_thresholds=custom_metrics_thresholds,
        wml_space_id=WML_SPACE_ID,
        apikey=API_KEY,
        auth_url=IAM_URL,
    )

    print(json.dumps(custom_monitor_instance_details.to_dict(), indent=4))
