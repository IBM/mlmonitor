# SPDX-License-Identifier: Apache-2.0
import logging
import os
from os.path import dirname, abspath
from ibm_cloud_sdk_core.authenticators import (
    IAMAuthenticator,
    CloudPakForDataAuthenticator,
)
from mlmonitor.config import (
    get_env,
    get_connection_details,
    get_wkc_details,
    get_aws_credentials,
    get_wos_details,
    get_cp4d_on_prem_details,
    assign_verify_ssl,
)


PROJECT_ROOT = abspath(dirname(dirname(__file__)))
DATA_ROOT = f"{PROJECT_ROOT}/datasets"
MODEL_ROOT = f"{PROJECT_ROOT}/models"
FIGURES_ROOT = f"{PROJECT_ROOT}/figures"
IAM_URL = "https://iam.bluemix.net/oidc/token"

ENV = get_env()
API_KEY, AUTH_ENDPOINT = get_connection_details(env=ENV, iam_url=IAM_URL)
MODEL_ENTRY_ID, CATALOG_ID = get_wkc_details(env=ENV)

key, secret, region, ROLE = get_aws_credentials()
aws_credentials = {
    "aws_access_key_id": key,
    "aws_secret_access_key": secret,
    "region_name": region,
}

WOS_URL, WOS_INSTANCE_ID = get_wos_details(env=ENV)
CP4D_VERSION, USERNAME = get_cp4d_on_prem_details(env=ENV)


if ENV == "saas":
    authenticator = IAMAuthenticator(apikey=API_KEY)
    VERIFY_CP4D_SSL = assign_verify_ssl(default_value=True)
elif ENV == "prem":
    authenticator = CloudPakForDataAuthenticator(
        url=AUTH_ENDPOINT,
        username=USERNAME,
        apikey=API_KEY,
        disable_ssl_verification=False,
    )
    VERIFY_CP4D_SSL = assign_verify_ssl(default_value=False)

else:
    raise ValueError(
        f"ENV set to '{ENV}'.Value should be set to 'saas' (IBM Cloud) or 'prem' (On premise cluster)"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(funcName)s -%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", logging.INFO)))
logger.debug(f"ENV set to {ENV}")
