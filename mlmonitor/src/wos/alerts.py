# SPDX-License-Identifier: Apache-2.0
import logging
from datetime import datetime, timedelta
import json

from mlmonitor.src.utils.utils import parse_args, build_aws_model_data, read_model_data
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.monitors import get_exising_monitors
from mlmonitor.src import PROJECT_ROOT, logger


def collect_alerts(
    deployment_name: str, monitor_types: tuple, output_model_data: str = PROJECT_ROOT
) -> None:
    """Trigger a new evaluation of the existing monitors listed in <monitor_types> for
    the deployed Sagemaker Endpoint with name <deployment_name>
    Parameters
    ----------
    deployment_name : str
       this indicates the Endpoint Name in Sagemaker for which an OpenScale subscription should be created
    monitor_types: str
        type(s) of monitor(s) to evaluate. Supported monitors are ['quality','fairness','drift','mrm']
    data_path : str
       location of dataset to be fetched to get scoring request samples
    output_model_data : str
       directory output Path where dictionary with model details in model_data.json file should be written
    Returns
    -------
    """

    if not deployment_name:
        deployment_info = read_model_data(model_dir=output_model_data)
        deployment_name = deployment_info.get("model_endpoint")

    else:
        deployment_info = build_aws_model_data(
            wos_client=wos_client, deployment_name=deployment_name
        )

    subscription_id = deployment_info.get("subscription_id")

    logger.debug(
        f"\nDeployment Name {deployment_name}\n" f"subscription_id {subscription_id}\n"
    )

    existing_monitors = get_exising_monitors(
        wos_client=wos_client, subscription_id=subscription_id
    )
    logger.debug(f"existing monitors\n{json.dumps(existing_monitors,indent=4)}")

    current_time = datetime.now()
    for monitor_type in monitor_types:
        if monitor_id := existing_monitors.get(monitor_type):
            logging.debug(f"Evaluate {monitor_type} Monitor ID {monitor_id}")
            # evaluate(wos_client=wos_client, monitor_instance_id=monitor_id, monitor_type=monitor_type)

            metrics_count = wos_client.monitor_instances.get_metrics_count(
                monitor_instance_id=monitor_id,
                start=(current_time - timedelta(days=2)),
                end=(current_time + timedelta(days=2)),
            )

            metrics = wos_client.monitor_instances.get_metrics(
                monitor_instance_id=monitor_id,
                start=(current_time - timedelta(days=2)),
                end=(current_time + timedelta(days=2)),
            )

            print(metrics.result.to_dict())

            logging.debug(
                f"metrics_count \n{json.dumps(metrics_count.result.to_dict(),indent=4)}"
            )


if __name__ == "__main__":
    args = parse_args()

    monitor_types = tuple(
        x
        for x, y in zip(
            ["quality", "fairness", "drift", "mrm"],
            [
                args.wos_evaluate_quality,
                args.wos_evaluate_fairness,
                args.wos_evaluate_drift,
                args.wos_evaluate_mrm,
            ],
        )
        if y
    )
    logger.info(f"Collect Alerts {monitor_types}...")
    collect_alerts(
        deployment_name=args.deployment_name,
        monitor_types=monitor_types,
        output_model_data=PROJECT_ROOT,
    )
