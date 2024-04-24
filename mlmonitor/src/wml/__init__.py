# SPDX-License-Identifier: Apache-2.0
from ibm_watson_machine_learning import APIClient as WML_APIClient

from mlmonitor.src import API_KEY, ENV, logger
from mlmonitor.config import get_wml_details

WML_INSTANCE_ID, WML_URL, WML_SPACE_ID = get_wml_details(env=ENV)
WML_CREDENTIALS = {"url": WML_URL}

SUPPORTED_WML_RUNTIMES = {
    "runtime-22.1-py3.10",
    "runtime-22.2-py3.10",
    "spark-mllib_3.3",
    "runtime-23.1-py3.10",
    "tensorflow_rt22.1-py3.9",
    "tensorflow_rt22.1-py3.10",
}

if ENV == "prem":
    from mlmonitor.src import CP4D_VERSION, USERNAME

    WML_CREDENTIALS["username"] = USERNAME
    # WML_CREDENTIALS["password"] = API_KEY
    WML_CREDENTIALS["apikey"] = API_KEY
    WML_CREDENTIALS["instance_id"] = WML_INSTANCE_ID
    WML_CREDENTIALS["version"] = CP4D_VERSION
elif ENV == "saas":
    WML_CREDENTIALS["apikey"] = API_KEY

try:
    logger.debug("Instantiate WML Client")
    wml_client = WML_APIClient(wml_credentials=WML_CREDENTIALS)
    wml_client.set.default_space(WML_SPACE_ID)
except Exception as e:
    wml_client = None
    logger.warning(f"Error to instantiate WML Client : {e}")
