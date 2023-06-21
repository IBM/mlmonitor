# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale.supporting_classes.enums import TargetTypes
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import Target

from mlmonitor.src.wos.monitors import get_monitor_id_by_subscription
from mlmonitor.src.wos import wos_client
from mlmonitor.src import logger
from mlmonitor.src.model.config import ModelConfig
from mlmonitor.src.wos.data_mart import get_datamart_ids
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment


def configure_explain(
    model_config: ModelConfig,
    deployment_name: str,
    keep_wos_monitor: bool = True,
    data_mart_id: str = None,
) -> dict:
    """

    - Finds the existing WOS subscription for a given deployment, if it exists.
    - Finds the existing Explainability Monitor instance for a given deployment, if it exists.
    - Re-Create or Create Explainability Monitor instance for WOS subscription found
    - parameters specified in model_config.explain_monitor_parameters will be used

    :param model_config: ModelConfig: Configuration parameters for the explain monitor
    :param deployment_name:str: Identify the deployment name
    :param keep_wos_monitor:bool=True: Delete the monitor instance if it already exists and value set to False
    :param data_mart_id:str=None: Specify the datamart to be used will be fetched if not specified
    :return: Explainability monitor instance id created or retrieved
    """

    subscription_ids = get_subscription_id_by_deployment(
        deployment_name=deployment_name, wos_client=wos_client
    )

    if len(subscription_ids) != 1:
        raise ValueError(
            f"{deployment_name} should have exactly one subscription ID => {len(subscription_ids)} found"
        )

    if not data_mart_id:
        data_marts = get_datamart_ids(wos_client=wos_client)

        if len(data_marts) != 1:
            raise ValueError(f"Please Specify datamart to use among {data_marts}")

        data_mart_id = data_marts[0]

    subscription_id = subscription_ids[0]

    if model_config.explain_monitor_enabled:

        # Find Monitors in place for a given SUBSCRIPTION_ID
        explain_monitor_instance_id = get_monitor_id_by_subscription(
            wos_client=wos_client,
            subscription_id=subscription_id,
            monitor_type="explainability",
        )

        if not keep_wos_monitor and explain_monitor_instance_id:
            wos_client.monitor_instances.delete(
                monitor_instance_id=explain_monitor_instance_id, background_mode=False
            )
            explain_monitor_instance_id = None

        if not explain_monitor_instance_id:

            explain_monitor_details = wos_client.monitor_instances.create(
                data_mart_id=data_mart_id,
                background_mode=False,
                monitor_definition_id=wos_client.monitor_definitions.MONITORS.EXPLAINABILITY.ID,
                target=Target(
                    target_type=TargetTypes.SUBSCRIPTION, target_id=subscription_id
                ),
                parameters={"enabled": model_config.explain_monitor_enabled},
            ).result

            explain_monitor_instance_id = explain_monitor_details.metadata.id

            logger.debug(f"explain Monitor ID Created [{explain_monitor_details}]")
        else:
            logger.warning(
                f"Explainability Monitor {explain_monitor_instance_id} Already exists"
            )
        return explain_monitor_instance_id

    else:
        logger.warning("Explain Monitor not Enabled in Configuration")
