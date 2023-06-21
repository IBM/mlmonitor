# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timedelta
import pytz
import json
from ibm_watson_openscale import APIClient as WOS_APIClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlmonitor.src.utils.utils import parse_args
from mlmonitor.src import logger
from mlmonitor.src.wos.monitors import get_exising_monitors
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment


def get_alerts(
    limits: list,
    measurements_dates: list,
    metrics: list,
    df_metrics: pd.DataFrame,
    deployment_name: str,
    alert_type: str = "quality",
    metric_name: str = "area_under_roc",
    limit_type: str = "lower",
) -> tuple:
    assert limit_type in {"upper", "lower"}

    df_limits = pd.DataFrame(
        data=np.array(limits).T, index=measurements_dates, columns=metrics
    )
    df_limits.index = pd.to_datetime(df_limits.index)
    logger.debug(f"\n[get_alerts] limits dataframe \n{df_limits}")

    # keep thresholds set per date per metric
    metrics_with_threshold_set = list(df_limits.dropna(axis=1, how="all").columns)
    model_metrics = df_metrics.loc[:, metrics_with_threshold_set].dropna().round(2)
    logger.info(f"\n[get_alerts] model_metrics dataframe \n{model_metrics}")
    # 	                                    drift_magnitude	data_drift_magnitude
    # 2022-10-06 03:41:59.376000+00:00	           0.031429	    0.15
    # 2022-10-06 03:45:28.665000+00:00	           0.021429	    0.16

    thresholds = df_limits.dropna(axis=1, how="all").loc[model_metrics.index]
    logger.info(f"\n[get_alerts] thresholds dataframe \n{thresholds}")
    # 	                              drift_magnitude	data_drift_magnitude
    # 2022-10-06 03:41:59.376000+00:00	0.1	                    0.1
    # 2022-10-06 03:45:28.665000+00:00	0.1	                    0.1

    metrics_minus_thresholds = np.subtract(model_metrics, thresholds)
    logger.debug(
        f"\n[get_alerts] metrics_minus_thresholds dataframe \n{metrics_minus_thresholds}"
    )
    if metrics_minus_thresholds.empty:
        return ()
    # for col in metrics_minus_thresholds:
    # !TODO support all metrics at once
    metrics_minus_thresholds["alert"] = (
        np.where(metrics_minus_thresholds[metric_name] > 0, True, False)
        if limit_type == "upper"
        else np.where(metrics_minus_thresholds[metric_name] < 0, True, False)
    )
    metrics_minus_thresholds[metric_name] = (
        metrics_minus_thresholds[metric_name].astype(float).round(3)
    )
    metrics_minus_thresholds["alert_type"] = alert_type
    metrics_minus_thresholds["limit_type"] = limit_type
    metrics_minus_thresholds["endpoint_name"] = deployment_name

    # Filter Only Alerts and metric name
    alerts_for_metric = metrics_minus_thresholds.loc[
        metrics_minus_thresholds["alert"],
        [metric_name, "alert_type", "endpoint_name"],
    ].rename(columns={metric_name: "threshold_violation"})
    # Merge with thresholds
    alerts_for_metric = alerts_for_metric.merge(
        thresholds.loc[:, [metric_name]], left_index=True, right_index=True
    ).rename(columns={metric_name: "threshold"})
    # Merge with model_metrics
    alerts_for_metric = alerts_for_metric.merge(
        model_metrics.loc[:, [metric_name]], left_index=True, right_index=True
    ).rename(columns={metric_name: "metric_value"})
    alerts_for_metric["metric_name"] = metric_name
    alerts_for_metric = alerts_for_metric.reset_index().rename(
        columns={"index": "alert_date"}
    )
    alerts_for_metric["alert_date"] = alerts_for_metric["alert_date"].astype(str)
    alerts_for_metric["alert_status"] = "new"
    logger.debug(
        f"\n[get_alerts] alerts_for_metric (filtered alerts) dataframe \n{alerts_for_metric}"
    )
    return alerts_for_metric.sort_index(ascending=True).to_dict(orient="records")


def plot_metrics(df_metrics: pd.DataFrame) -> None:
    df_ = df_metrics.dropna()

    fig, axes = plt.subplots(len(df_metrics.columns), 1, sharex=True, figsize=(10, 20))
    color = list(plt.rcParams["axes.prop_cycle"])

    for i, col in enumerate(df_.columns):
        axes[i].plot(df_[col], color=color[i % (len(color) - 1)]["color"])

        axes[i].label_outer()
        axes[i].set_title(col)

    # plt.tight_layout()
    plt.xticks(rotation=90)
    plt.show()


def collect_alerts(
    alert_type: str,
    metric_name: str,
    limit_type: str,
    deployment_name: str,
    start_date: str = None,
    end_date: str = None,
    interval: str = "hour",
    wos_client: WOS_APIClient = None,
) -> tuple:
    """Collect <alert_type> Alerts for Endpoint with name <deployment_name> between <start_date> and <end_date>
    Parameters
    ----------
    alert_type : str
       this indicates the alert type
    metric_name : str
       monitored metric type e.g accuracy,precision,recall
    limit_type: str
        limit type applied for this metric_type lower,upper
    interval:
        time interval for metrics collection e.g 'minute','hour','day'
    deployment_name : str
       this indicates the Endpoint Name in Sagemaker for which an OpenScale subscription should be created
    start_date : str
       start date of alert collection
    end_date : str
       end date of alert collection
    wos_client : WOS_APIClient
       Watson OpenScale client instatiated
    Returns
    -------
    alerts : tuple
       tuple of alerts for a given alert_type,endpoint_name

        ({'alert_date': '',
        'threshold_violation': 0.01,
        'alert_type': 'drift',
        'endpoint_name': '<endpoint name>',
        'threshold': 0.1,
        'metric_value': 0.11,
        'metric_name': 'data_drift_magnitude',
        'alert_status': 'new'},{...},{....},)
    """
    current_time = datetime.utcnow().replace(tzinfo=pytz.utc)
    start_date = (
        datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%f%z")
        if start_date
        else current_time - timedelta(minutes=60)
    )
    end_date = (
        datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S.%f%z")
        if end_date
        else current_time
    )
    logger.info(
        f'Collecting Alerts between  "{start_date}" and  {end_date}  for {alert_type}:{metric_name}'
    )

    #############################################################
    # FETCH METRICS USING WOS SDK BETWEEN START AND END DATE    #
    #############################################################
    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=deployment_name
    )

    if len(subscription_ids) != 1:
        raise ValueError(
            f"{deployment_name} should have exactly one subscription ID => {len(subscription_ids)} found"
        )

    subscription_id = subscription_ids[0]

    logger.debug(
        f"Deployment Name {deployment_name}\n" f"subscription_id {subscription_id}\n"
    )

    existing_monitors = get_exising_monitors(
        wos_client=wos_client, subscription_id=subscription_id
    )
    monitor_id = existing_monitors.get(alert_type)

    wos_metrics = wos_client.monitor_instances.get_metrics(
        monitor_instance_id=monitor_id,
        start=start_date,
        end=end_date,
        interval=interval,
    )

    # collected_metrics_example = \
    # {"start": "2022-10-06T03:40:00Z",
    #  "end": "2022-10-06T03:46:00Z",
    #  "interval": "minute",
    #  "monitor_definition_id": "drift",
    #  "groups": [{"tags": [],
    #              "metrics": [{"id": "drift_magnitude", "lower_limit": [None, None, None, None, None, None],
    #                           "upper_limit": [None, 0.1, None, None, None, 0.1],
    #                           "last": {"value": [None, 0.03142857142857136, None, None, None, 0.021428571428571463],
    #                                    "measurement_id": [None, "79238f98-699b-4a84-8d04-025a615c26c2", None, None,
    #                                                       None,
    #                                                       "dafbb05d-0097-4f3c-9571-62bb25dfe308"]}},
    #                          {"id": "predicted_accuracy",
    #                           "lower_limit": [None, None, None, None, None, None],
    #                           "upper_limit": [None, None, None, None, None, None],
    #                           "last": {"value": [None, 0.93, None, None, None, 0.94],
    #                                    "measurement_id": [None, "79238f98-699b-4a84-8d04-025a615c26c2",
    #                                                       None, None, None, "dafbb05d-0097-4f3c-9571-62bb25dfe308"]}},
    #                          {"id": "data_drift_magnitude",
    #                           "lower_limit": [None, None, None, None, None, None],
    #                           "upper_limit": [None, 0.1, None, None, None, 0.1],
    #                           "last": {"value": [None, 0.15, None, None, None, 0.16],
    #                                    "measurement_id": [None, "79238f98-699b-4a84-8d04-025a615c26c2", None, None,
    #                                                       None,
    #                                                       "dafbb05d-0097-4f3c-9571-62bb25dfe308"]}}]}]}

    metrics_groups = wos_metrics.result.to_dict().get("groups")
    logger.info(f"Collecting metrics with {len(metrics_groups)} groups")

    if len(metrics_groups) <= 0:
        return ()
    group_id = 0
    metric_values = metrics_groups[group_id].get("metrics")
    # get metric names for for alert_type e.g drift has drift_magnitude,predicted_accuracy and data_drift_magnitude
    metrics = [x.get("id") for x in metric_values if len(metrics_groups) > 0]
    logger.debug(f"{len(metrics)} Existing metrics for {alert_type} are {metrics}")
    # get limits set a a specific date for each metric
    lower_limits = [x.get("lower_limit") for x in metric_values]
    upper_limits = [x.get("upper_limit") for x in metric_values]

    # get metric values
    values = [x.get("last").get("value") for x in metric_values]
    logger.debug(f"group {group_id} with {len(values)} values:\n{values}")
    # get measurements identifiers
    measurements = [x.get("last").get("measurement_id") for x in metric_values][0]
    measurements_dates = [
        wos_client.monitor_instances.get_measurement_details(
            monitor_instance_id=monitor_id, measurement_id=x
        )
        .result.to_dict()
        .get("metadata")
        .get("created_at")
        if x
        else None
        for x in measurements
    ]

    # last_measurement_date = datetime.strptime(measurements_dates[-1], "%Y-%m-%dT%H:%M:%S.%f%z")
    # !TODO cleanup
    # ago = (current_time - last_measurement_date)

    # logger.debug(f"last {alert_type}:{metric_name} measurement was {last_measurement_date} : {ago}")

    df_metrics = pd.DataFrame(
        data=np.array(values).T, index=measurements_dates, columns=metrics
    )
    df_metrics.index = pd.to_datetime(df_metrics.index)

    logger.debug(f"metrics dataframe \n\n{df_metrics}")

    plot_metrics(df_metrics)

    ##############################################
    # FILTERING ALERTS FROM METRICS DATAFRAME    #
    ##############################################

    if limit_type == "lower":
        limits = lower_limits

    elif limit_type == "upper":
        limits = upper_limits
    else:
        raise ValueError("limit_type should be lower or upper")

    return get_alerts(
        limits=limits,
        measurements_dates=measurements_dates,
        metrics=metrics,
        deployment_name=deployment_name,
        df_metrics=df_metrics,
        alert_type=alert_type,
        metric_name=metric_name,
        limit_type=limit_type,
    )


if __name__ == "__main__":

    import os
    from mlmonitor.src.wos import wos_client
    from mlmonitor.src.utils.utils import send_alert

    args = parse_args()

    with open(
        os.environ.get("MONITOR_CONFIG_FILE", "../../../credentials.cfg"), "r"
    ) as f:
        mlmonitor_credentials = json.load(f)

    username = mlmonitor_credentials.get("mlops_orchestrator").get("username")
    password = mlmonitor_credentials.get("mlops_orchestrator").get("password")
    url = mlmonitor_credentials.get("mlops_orchestrator").get("scoring_url")

    deployment_name = args.deployment_name

    alert_type = "drift"
    metric_name = "data_drift_magnitude"
    limit_type = "upper"

    # alert_type = 'quality'
    # metric_name = 'area_under_roc'
    # limit_type = 'lower'

    # data but no alert
    # start_date = '2022-11-25T05:00:00.000000+00:00'
    # end_date = '2022-11-25T05:21:00.000000+00:00'

    # data drift alert
    start_date = "2022-11-25T05:20:00.000000+00:00"
    end_date = "2022-11-25T05:30:00.000000+00:00"

    # No Date
    # start_date = None
    # end_date = None
    interval = "minute"

    current_time = datetime.utcnow().replace(tzinfo=pytz.utc)
    print(current_time)

    alerts = collect_alerts(
        alert_type=alert_type,
        limit_type=limit_type,
        metric_name=metric_name,
        deployment_name=deployment_name,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        wos_client=wos_client,
    )

    print(alerts)
    for new_alert in alerts:
        print(new_alert)
        resp = send_alert(
            url=url, username=username, password=password, request_body=new_alert
        )
        print(resp)
