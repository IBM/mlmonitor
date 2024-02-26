# SPDX-License-Identifier: Apache-2.0
from mlmonitor.src.wos.integated_system import (
    get_integrated_system_by_provider_name,
    create_custom_metric_definitions,
)
from mlmonitor.src.wos.monitors import (
    get_monitor_id_by_subscription,
    get_exising_monitors,
)

from ibm_watson_openscale.base_classes.watson_open_scale_v2 import IntegratedSystems
from ibm_watson_openscale import APIClient
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import (
    MetricThresholdOverride,
)
from ibm_watson_openscale.supporting_classes.enums import MetricThresholdTypes
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import Target
from ibm_watson_openscale.supporting_classes.enums import TargetTypes
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import (
    MonitorInstanceSchedule,
    ScheduleStartTime,
)


def get_custom_monitor_definition(wos_client: APIClient, monitor_name: str):
    filtered_monitors = [
        definition
        for definition in wos_client.monitor_definitions.list().result.monitor_definitions
        if definition.entity.name == monitor_name
    ]
    if len(filtered_monitors) == 1:
        return filtered_monitors[0]
    elif len(filtered_monitors) == 0:
        return None
    else:
        raise ValueError(
            f"Number of monitors found for monitor name  {monitor_name} !=1 => {len(filtered_monitors)} "
        )


def get_custom_monitor_instance(
    wos_client: APIClient,
    data_mart_id: str,
    monitor_definition_id: str,
    subscription_id: str,
):
    monitor_instances = list(
        wos_client.monitor_instances.list(
            data_mart_id=data_mart_id,
            monitor_definition_id=monitor_definition_id,
            target_target_id=subscription_id,
        ).result.monitor_instances
    )
    if len(monitor_instances) == 1:
        return monitor_instances[0]
    elif len(monitor_instances) == 0:
        return None
    else:
        raise ValueError(
            f"Number of monitors instances found !=1 => {len(monitor_instances)} "
        )


def create_custom_monitor_definition(
    wos_client: APIClient,
    custom_monitor_name: str,
    custom_metrics_names: tuple,
    custom_metrics_thresholds: tuple,
    enable_schedule: bool = True,
    repeat_interval: int = 1,
    repeat_type: str = "hour",
    delay_unit: str = "minute",
    delay_time: int = 30,
):
    # check if the custom monitor definition already exists or not
    existing_definition = get_custom_monitor_definition(
        wos_client=wos_client, monitor_name=custom_monitor_name
    )

    start_time = ScheduleStartTime(
        type="relative", delay_unit=delay_unit, delay=delay_time
    )

    # if custom metric does not exist, then create a new one.
    if existing_definition is None:
        metrics, tags = create_custom_metric_definitions(
            custom_metrics_names=custom_metrics_names,
            custom_metrics_thresholds=custom_metrics_thresholds,
        )
        return (
            wos_client.monitor_definitions.add(
                name=custom_monitor_name,
                metrics=metrics,
                tags=tags,
                schedule=MonitorInstanceSchedule(
                    repeat_interval=repeat_interval,
                    repeat_unit=repeat_type,
                    start_time=start_time,
                ),
                background_mode=False,
            ).result
            if enable_schedule
            else wos_client.monitor_definitions.add(
                name=custom_monitor_name,
                metrics=metrics,
                tags=tags,
                background_mode=False,
            ).result
        )
    else:
        # otherwise, send the existing definition
        print(f"Existing Definition found with ID {existing_definition.metadata.id}")
        return existing_definition


def update_custom_monitor_instance(
    custom_monitor_instance_id: str,
    integrated_system_id: str,
    wos_client: APIClient,
    custom_metrics_wait_time: int = 300,
    max_records: int = None,
    min_records: int = None,
):
    payload = [
        {
            "op": "replace",
            "path": "/parameters",
            "value": {
                "custom_metrics_provider_id": integrated_system_id,
                "custom_metrics_wait_time": custom_metrics_wait_time,
                "enable_custom_metric_runs": True,
            },
        }
    ]
    if max_records is not None:
        payload[0]["value"]["max_records"] = max_records
    if min_records is not None:
        payload[0]["value"]["min_records"] = min_records

    response = wos_client.monitor_instances.update(
        custom_monitor_instance_id, payload, update_metadata_only=True
    )
    result = response.result
    return result


def create_custom_monitor_instance(
    wos_client: APIClient,
    data_mart_id: str,
    integrated_system_id: str,
    monitor_definition_id: str,
    subscription_id: str,
    custom_metrics_names: tuple,
    custom_metrics_thresholds: tuple,
    custom_metrics_wait_time: int = 300,
    max_records: int = None,
    min_records: int = None,
):
    # Check if a custom monitor instance already exists
    existing_monitor_instance = get_custom_monitor_instance(
        wos_client=wos_client,
        data_mart_id=data_mart_id,
        monitor_definition_id=monitor_definition_id,
        subscription_id=subscription_id,
    )

    # If it does not exist, then create one
    if existing_monitor_instance is None:
        target = Target(target_type=TargetTypes.SUBSCRIPTION, target_id=subscription_id)
        parameters = {
            "custom_metrics_provider_id": integrated_system_id,
            "custom_metrics_wait_time": custom_metrics_wait_time,
            "enable_custom_metric_runs": True,
        }
        if max_records is not None:
            parameters["max_records"] = max_records
        if min_records is not None:
            parameters["min_records"] = min_records

        # Update your custom monitor metric ids in the below thresholds to update the default value
        thresholds = [
            MetricThresholdOverride(
                metric_id=name,
                type=MetricThresholdTypes.UPPER_LIMIT,
                value=threshold,  # LOWER_LIMIT
            )
            for name, threshold in zip(custom_metrics_names, custom_metrics_thresholds)
        ]

        # create the custom monitor instance id here.
        return wos_client.monitor_instances.create(
            data_mart_id=data_mart_id,
            background_mode=False,
            monitor_definition_id=monitor_definition_id,
            target=target,
            parameters=parameters,
            thresholds=thresholds,
        ).result
    else:
        # otherwise, update the existing one with the latest integrated system details.
        instance_id = existing_monitor_instance.metadata.id
        return update_custom_monitor_instance(
            wos_client=wos_client,
            custom_monitor_instance_id=instance_id,
            custom_metrics_wait_time=custom_metrics_wait_time,
            min_records=min_records,
            max_records=max_records,
            integrated_system_id=integrated_system_id,
        )


def cleanup_custom_monitor(
    wos_client: APIClient,
    provider_name: str,
    custom_monitor_name: str,
    subscription_id: str,
    data_mart_id: str,
):
    integrated_system_ids = get_integrated_system_by_provider_name(
        wos_client, provider_name
    )

    if len(integrated_system_ids) == 1:
        integrated_system_id = integrated_system_ids[0]
        print(f"[Cleanup] Deleting Integrated System ID {integrated_system_id}")
        IntegratedSystems(wos_client).delete(integrated_system_id=integrated_system_id)

    if existing_definition := get_custom_monitor_definition(
        wos_client=wos_client, monitor_name=custom_monitor_name
    ):
        if existing_monitor_instance := get_custom_monitor_instance(
            wos_client=wos_client,
            data_mart_id=data_mart_id,
            monitor_definition_id=existing_definition.metadata.id,
            subscription_id=subscription_id,
        ):
            print(
                f"[Cleanup] Deleting Monitor Instance {existing_monitor_instance.metadata.id}"
            )
            wos_client.monitor_instances.delete(
                monitor_instance_id=existing_monitor_instance.metadata.id
            )

        print(
            f"[Cleanup] Deleting Monitor Definition {existing_definition.metadata.id}"
        )
        wos_client.monitor_definitions.delete(
            monitor_definition_id=existing_definition.metadata.id
        )

    print(custom_monitor_name.lower())
    wos_client.monitor_instances.show(limit=100)

    existing_monitors = get_exising_monitors(
        wos_client=wos_client, subscription_id=subscription_id
    )

    if custom_monitor_name in existing_monitors:
        monitor_id = get_monitor_id_by_subscription(
            wos_client=wos_client,
            subscription_id=subscription_id,
            monitor_type=custom_monitor_name.lower(),
        )
        wos_client.monitor_instances.delete(
            monitor_instance_id=monitor_id, background_mode=False
        )
