# SPDX-License-Identifier: Apache-2.0
from mlmonitor.src.utils.utils import parse_args
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wml import wml_client, WML_SPACE_ID
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.wos.custom_monitors import cleanup_custom_monitor
from mlmonitor.src.wos.data_mart import get_datamart_ids

import importlib

if __name__ == "__main__":

    args = parse_args()

    model_signature = importlib.import_module(
        f"mlmonitor.{args.source_dir}.model_signature"
    )
    custom_monitor = getattr(model_signature, "custom_monitor")

    wml_client.set.default_space(WML_SPACE_ID)

    # Model for which the Custom monitor is attached (via subscription)
    monitored_model = args.deployment_name

    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=monitored_model
    )

    data_marts = get_datamart_ids(wos_client=wos_client)
    data_mart_id = data_marts[0]

    if len(subscription_ids) == 1:
        subscription_id = subscription_ids[0]
    else:
        raise ValueError(f"No WOS subscription found for deployment {monitored_model}")

    # CUSTOM MONITOR SPECIFIC NAMES
    provider_name = custom_monitor.get("provider_name")
    # Name Displayed in WOS UI
    custom_monitor_name = custom_monitor.get("custom_monitor_name")

    cleanup_custom_monitor(
        wos_client=wos_client,
        provider_name=provider_name,
        custom_monitor_name=custom_monitor_name,
        subscription_id=subscription_id,
        data_mart_id=data_mart_id,
    )
