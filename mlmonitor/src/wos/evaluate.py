# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timedelta
import json

from mlmonitor.src.utils.utils import parse_args
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.monitors import get_exising_monitors, evaluate
from mlmonitor.src import logger
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment


def evaluate_monitor(deployment_name: str, monitor_types: tuple) -> None:
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

    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=deployment_name
    )

    if len(subscription_ids) != 1:
        raise ValueError(
            f"{deployment_name} should have exactly one subscription ID => {len(subscription_ids)} found"
        )

    subscription_id = subscription_ids[0]

    logger.debug(
        f"\nDeployment Name {deployment_name}\n" f"subscription_id {subscription_id}\n"
    )

    existing_monitors = get_exising_monitors(
        wos_client=wos_client, subscription_id=subscription_id
    )
    logger.debug(f"existing monitors\n{json.dumps(existing_monitors, indent=4)}")

    current_time = datetime.now()
    for monitor_type in monitor_types:
        assert monitor_type in set(
            list(existing_monitors.keys())
            + ["quality", "fairness", "explainability", "drift", "mrm"]
        )
        if (monitor_id := existing_monitors.get(monitor_type)) and monitor_type not in [
            "mrm",
            "explainability",
        ]:
            logger.info(f"Evaluate {monitor_type} Monitor ID {monitor_id}")
            evaluate(
                wos_client=wos_client,
                monitor_instance_id=monitor_id,
                monitor_type=monitor_type,
            )

            metrics_count = wos_client.monitor_instances.get_metrics_count(
                monitor_instance_id=monitor_id,
                start=(current_time - timedelta(days=2)),
                end=(current_time + timedelta(days=2)),
            )
            logger.debug(
                f"metrics_count \n{json.dumps(metrics_count.result.to_dict(), indent=4)}"
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
    logger.info(f"Running Monitors {monitor_types}...")
    evaluate_monitor(deployment_name=args.deployment_name, monitor_types=monitor_types)
