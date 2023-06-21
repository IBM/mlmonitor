# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale import APIClient
from datetime import datetime
import json
import requests
from mlmonitor.src import logger


def get_monitor_id_by_subscription(
    wos_client: APIClient, subscription_id: str, monitor_type: str = "quality"
):
    supported_monitors = [
        x.metadata.id
        for x in wos_client.monitor_definitions.list().result.monitor_definitions
    ]
    if monitor_type not in supported_monitors:
        raise ValueError(
            f"monitor_type must be in {supported_monitors} => {monitor_type} passed "
        )

    filtered_monitors = [
        mon.metadata.id
        for mon in wos_client.monitor_instances.list().result.monitor_instances
        if mon.entity.target.target_type == "subscription"
        and mon.entity.target.target_id == subscription_id
        and mon.entity.monitor_definition_id == monitor_type
    ]
    if len(filtered_monitors) == 1:
        return filtered_monitors[0]
    elif len(filtered_monitors) == 0:
        return None
    else:
        raise ValueError(
            f"Number of {monitor_type} monitors found for subscription_id  {subscription_id} !=1 => {len(filtered_monitors)} "
        )


def get_exising_monitors(wos_client: APIClient, subscription_id: str) -> dict:
    return {
        mon.entity.monitor_definition_id: mon.metadata.id
        for mon in wos_client.monitor_instances.list().result.monitor_instances
        if mon.entity.target.target_type == "subscription"
        and mon.entity.target.target_id == subscription_id
    }


def evaluate(wos_client: APIClient, monitor_type: str, monitor_instance_id: str):
    current_time = datetime.now()
    logger.info(
        f'Run {monitor_type} Monitor evaluation started at {current_time.strftime("%d/%m/%Y %H:%M:%S")}'
    )
    try:

        run_details = wos_client.monitor_instances.run(
            monitor_instance_id=monitor_instance_id, background_mode=False
        ).result

        logger.debug(f"[{monitor_type}] runs details:\n{run_details}")

        logger.info(
            f'{monitor_type} Monitor evaluation completed at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
        )
        runs = wos_client.monitor_instances.list_runs(
            monitor_instance_id=monitor_instance_id
        ).result.to_dict()

        logger.debug(
            f"[{monitor_type}] runs result:\n{json.dumps(runs, indent=4, sort_keys=True)}"
        )

    except requests.exceptions.ReadTimeout as e:
        logger.error(f"requests.exceptions.ReadTimeout {e}")
        logger.error(
            f'Run Quality Monitor evaluation Failed at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
        )
