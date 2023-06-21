# SPDX-License-Identifier: Apache-2.0
import datetime
import uuid
from typing import Tuple, Optional, Callable
import requests


def get_access_token_prem(username: str, apikey: str, url: str) -> str:
    """
    uses the apikey to get an access token from CP4D running on OCP.
    It returns a JSON object with the access token and expiration date.

    :param url:str CP4D base url
    :param username:dtr CP4D username
    :param apikey:str  CP4D API key
    :return: The access token for the IAM service
    """
    url = f"{url}/icp4d-api/v1/authorize"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {"username": username, "api_key": apikey}
    response = requests.post(url, headers=headers, json=payload, verify=False)

    if response.status_code != 200:
        raise ValueError(f"Failed to authenticate to CP4D SaaS {url} {response.text}")

    return response.json()["token"]


def get_access_token_saas(apikey: str) -> str:
    """
    uses the apikey to get an access token from the IAM service on IBM Cloud.
    It returns a JSON object with the access token and expiration date.

    :param apikey:str IBM Cloud API key
    :return: The access token for the IAM service
    """
    auth_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": "Basic Yng6Yng=",
    }
    data = {"apikey": apikey, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"}
    response = requests.post(
        "https://iam.ng.bluemix.net/identity/token", headers=auth_headers, data=data
    )

    if response.status_code != 200:
        raise ValueError(
            f"Failed to authenticate to on premise CP4D cluster {response.text}"
        )

    return response.json()["access_token"]


def get_feedback_dataset_id(
    access_token: str, data_mart_id: str, subscription_id: str, url: str
) -> str:
    """
    returns Watson OpenScale feedback dataset ID for a given subscription.

    :param access_token: Access the watson openscale apis
    :param data_mart_id: Specify the id of the openscale datamart
    :param subscription_id: Specify the subscription that feedback data will be inserted into
    :param url: Specify the url of the target openscale instance
    :return: The id of the feedback dataset
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    DATASETS_URL = f"{url}/openscale/{data_mart_id}/v2/data_sets?target.target_id={subscription_id}&target.target_type=subscription&type=feedback"
    response = requests.get(DATASETS_URL, headers=headers, verify=False)
    json_data = response.json()
    return (
        json_data["data_sets"][0]["metadata"]["id"]
        if "data_sets" in json_data and len(json_data["data_sets"]) > 0
        else None
    )


def get_feedback_data(
    access_token: str,
    data_mart_id: str,
    feedback_dataset_id: str,
    url: str,
    limit: int = 100,
) -> dict:
    """
    retrieves the feedback data from Watson OpenScale feedback dataset.

    :param limit: int=100
    :param access_token: Access token
    :param data_mart_id: Watson OpenScale Data Mart ID
    :param feedback_dataset_id: Watson OpenScale feedback_dataset_id associated with subscription
    :param url: Watson OpenScale url
    :param limit:int: Limit the number of feedback records returned
    :return: A list of records from the feedback dataset
    """
    if feedback_dataset_id:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        DATASETS_STORE_RECORDS_URL = f"{url}/openscale/{data_mart_id}/v2/data_sets/{feedback_dataset_id}/records?limit={limit}&format=list"
        response = requests.get(
            DATASETS_STORE_RECORDS_URL, headers=headers, verify=False
        )
        return response.json()


def get_error_json(error_message: str) -> dict:
    """
    JSON object that is used to communicate errors back to the client.
    The trace field is a unique identifier for this error message, and the errors array contains an error code and
    a description of what went wrong.

    :param error_message: Return a custom error message to the user
    :return: A dictionary code and message
    """
    trace = str(uuid.uuid4())
    return {
        "trace": trace,
        "errors": [
            {
                "code": "custom_metrics_error_code",
                "message": str(error_message),
            }
        ],
    }


def get_patch_request_field(
    base_path: str, field_name: str, field_value: str, op_name="replace"
) -> dict:
    return {
        "op": op_name,
        "path": f"{base_path}/{field_name}",
        "value": field_value,
    }


# Update the run status to Finished in the Monitor Run
def update_monitor_run_status(
    base_url: str,
    access_token: str,
    custom_monitor_instance_id: str,
    run_id: str,
    status: str,
    error_msg: Optional[str] = None,
) -> Tuple[int, str]:
    """
    updates the status of a monitor run.

    :param base_url:str:monitor  url
    :param access_token:str: Pass the access token to the function
    :param custom_monitor_instance_id:str: Identify the custom monitor instance that is being updated
    :param run_id:str: custom monitor evaluation run identifier
    :param status:str: status of the ongoing custom monitor run
    :param error_msg:str=None: Pass in an error message if the status is set to error;
    :return: A tuple of the status code and the response from the patch request
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    monitor_run_url = (
        f"{base_url}/v2/monitor_instances/{custom_monitor_instance_id}/runs/{run_id}"
    )
    completed_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    base_path = "/status"

    patch_payload = [
        get_patch_request_field(base_path, "state", status),
        get_patch_request_field(base_path, "completed_at", completed_timestamp),
    ]
    if error_msg:
        error_json = get_error_json(error_msg)
        patch_payload.append(get_patch_request_field(base_path, "failure", error_json))

    response = requests.patch(
        monitor_run_url, headers=headers, json=patch_payload, verify=False
    )
    monitor_run_response = response.json()
    return response.status_code, monitor_run_response


# Publishes the Custom Metrics to OpenScale
def publish_metrics(
    url: str,
    access_token: str,
    data_mart_id: str,
    subscription_id: str,
    get_metrics_fn: Callable,
    custom_monitor_id: str,
    custom_monitor_instance_id: str,
    custom_monitoring_run_id: str,
    timestamp: str,
) -> Tuple[int, dict]:
    """
    publishes custom metrics to Watson OpenScale monitor as part of an ongoing evaluation.

    :param url:str: WOS url
    :param access_token:str: Access token for the Watson Openscale access
    :param data_mart_id:str: Specify the id of the openscale data_mart to which the custom monitoring instance will be published
    :param subscription_id:str: Identify the subscription that we want to publish metrics for
    :param get_metrics_fn:Callable: Specify the function that is used to retrieve the metrics
    :param custom_monitor_id:str: Specify the custom monitor identifier created in Watson Openscale subscription
    :param custom_monitor_instance_id:str: Specify the custom monitoring instance id
    :param custom_monitoring_run_id:str: Custom monitor evaluation run identifier
    :param timestamp:str: Specify the time at which the measurement is taken
    :return: The status code of the response and the payload
    """
    base_url = f"{url}/openscale/{data_mart_id}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    # Generate a monitoring run id, where the publishing happens against this run id
    custom_metrics = get_metrics_fn(access_token, data_mart_id, subscription_id, url)

    measurements_payload = [
        {
            "timestamp": timestamp,
            "run_id": custom_monitoring_run_id,
            "metrics": [custom_metrics],
        }
    ]

    measurements_url = (
        f"{base_url}/v2/monitor_instances/{custom_monitor_instance_id}/measurements"
    )
    response = requests.post(
        measurements_url, headers=headers, json=measurements_payload, verify=False
    )
    published_measurement = response.json()

    return response.status_code, published_measurement


def publish(
    input_data: dict,
    url: str,
    apikey: str,
    get_metrics_fn: Callable,
    env: str = "saas",
    username: str = "admin",
) -> dict:
    """
    publish the metrics of a custom monitor run to Watson OpenScale.
    implementation of CustomMonitors Workflow

    Custom Monitor (WOS)                                         Custom Metrics Provider (WML)
                               --------------------------->  (run_id, subscription_id, datamart_id)
                                      Trigger Evaluation

                               <----------------------------
                                    Fetch Scoring Data (get datasets/records)

                               <----------------------------
                                    publish Metrics  (post measurements)

                               <---------------------------- (run_id)
                                    Update run status (patch runs)

    :param input_data:dict: payload from the trigger evaluation call sent by Watson Openscale
    :param url:str: Specify the url of the deployment of wml function
    :param apikey:str: IAM API Key
    :param get_metrics_fn:Callable: customer business logic function to compute custom metrics
    :param env:str = type of environment where custom metrics provider is deployed 'saas' or 'prem'
    :param username:str = username of CP4D user only used if 'prem' environment is selected
    :return: Payload response with metrics
    """
    # input_data received from (1) Trigger Evaluation
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    payload = input_data.get("input_data")[0].get("values")
    data_mart_id = payload["data_mart_id"]
    subscription_id = payload["subscription_id"]
    custom_monitor_id = payload["custom_monitor_id"]
    custom_monitor_instance_id = payload["custom_monitor_instance_id"]
    custom_monitor_run_id = payload["custom_monitor_run_id"]
    test = bool(payload.get("test"))

    base_url = f"{url}/openscale/{data_mart_id}"

    try:
        if env == "prem":
            access_token = get_access_token_prem(
                apikey=apikey, username=username, url=url
            )
        elif env == "saas":
            access_token = get_access_token_saas(apikey=apikey)
        else:
            raise ValueError("env parameter should be saas or prem")
    except Exception as ex:
        return {"predictions": [], "errors": str(ex)}

    if test:
        # used to query directly WML python function deployment
        metrics = get_metrics_fn(access_token, data_mart_id, subscription_id, url)
        return {
            "predictions": [
                {"fields": list(metrics.keys()), "values": list(metrics.values())}
            ]
        }

    published_measurements = []
    error_msgs = []
    run_status = "finished"

    try:
        error_msg = None
        # GET SCORING DATA (2) COMPUTE METRICS + PUBLISH METRICS on `measurements` endpoint (3)
        status_code, published_measurement = publish_metrics(
            url,
            access_token,
            data_mart_id,
            subscription_id,
            get_metrics_fn,
            custom_monitor_id,
            custom_monitor_instance_id,
            custom_monitor_run_id,
            timestamp,
        )

        if int(status_code) in {200, 201, 202}:
            published_measurements.append(published_measurement)
        else:
            run_status = "error"
            error_msg = published_measurement
            error_msgs.append(error_msg)
        # UPDATE STATUS (4)
        status_code, response = update_monitor_run_status(
            base_url,
            access_token,
            custom_monitor_instance_id,
            custom_monitor_run_id,
            run_status,
            error_msg,
        )

        if int(status_code) not in {200, 201, 202}:
            error_msgs.append(response)

    except Exception as ex:
        error_msgs.append(str(ex))
    return (
        {"predictions": [], "errors": error_msgs}
        if error_msgs
        else {"predictions": [{"values": published_measurements}]}
    )
