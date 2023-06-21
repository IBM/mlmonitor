# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale import APIClient as WOS_APIClient
from mlmonitor.src import authenticator, logger
from mlmonitor.src import WOS_URL, WOS_INSTANCE_ID

# in some account , instantiating wos client without service_instance_id failed with
# AuthorizationError: You are not authorized to access AI OpenScale instance None
# service_instance_id was added to fix this issue and is optional
try:
    logger.debug("Instantiate WOS Client")
    wos_client = WOS_APIClient(
        authenticator=authenticator,
        service_url=WOS_URL,
        service_instance_id=WOS_INSTANCE_ID,
    )
except Exception as e:
    wos_client = None
    logger.warning(f"Error to instantiate WOS Client : {e}")
