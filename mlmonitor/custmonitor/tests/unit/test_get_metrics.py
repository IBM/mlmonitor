# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import unittest
from unittest.mock import patch
from unittest import TestCase
from ibm_watson_openscale import APIClient as WOS_APIClient
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from custmonitor.metricsprovider.helpers import get_access_token_cloud

from mlmonitor.src import get_connection_details, WOS_URL
from mlmonitor.src.wos.data_mart import get_datamart_ids
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment


class TestGetMetrics(TestCase):
    @patch.dict(
        "os.environ",
        {
            "MONITOR_CONFIG_FILE": str(
                Path(__file__).parent / "config" / "credentials_gcr.cfg"
            )
        },
        clear=True,
    )
    def test_get_metrics_gcr(self):
        from custmonitor.metrics.credit_risk import get_metrics

        API_KEY, AUTH_ENDPOINT = get_connection_details()
        wos_client = WOS_APIClient(authenticator=IAMAuthenticator(apikey=API_KEY))
        monitored_deployment = "GCR_Deploy_Scikit_demo"
        token = get_access_token_cloud(apikey=API_KEY)

        subscription_ids = get_subscription_id_by_deployment(
            wos_client=wos_client, deployment_name=monitored_deployment
        )
        data_marts = get_datamart_ids(wos_client=wos_client)
        data_mart_id = data_marts[0]
        if len(subscription_ids) == 1:
            subscription_id = subscription_ids[0]
        else:
            raise ValueError(
                f"No WOS subscription found for deployment {monitored_deployment}"
            )
        res = get_metrics(token, data_mart_id, subscription_id, WOS_URL)
        self.assertIsInstance(res, dict)
        print(res)

    @patch.dict(
        "os.environ",
        {
            "MONITOR_CONFIG_FILE": str(
                Path(__file__).parent / "config" / "credentials_churn.cfg"
            )
        },
        clear=True,
    )
    def test_get_metrics_churn(self):
        from custmonitor.metrics.customer_churn import get_metrics

        API_KEY, AUTH_ENDPOINT = get_connection_details()
        wos_client = WOS_APIClient(authenticator=IAMAuthenticator(apikey=API_KEY))
        monitored_deployment = "sm-cc-xgboost-2022-10-13-14-36-11-945"
        token = get_access_token_cloud(apikey=API_KEY)

        subscription_ids = get_subscription_id_by_deployment(
            wos_client=wos_client, deployment_name=monitored_deployment
        )
        data_marts = get_datamart_ids(wos_client=wos_client)
        data_mart_id = data_marts[0]
        if len(subscription_ids) == 1:
            subscription_id = subscription_ids[0]
        else:
            raise ValueError(
                f"No WOS subscription found for deployment {monitored_deployment}"
            )
        res = get_metrics(token, data_mart_id, subscription_id, WOS_URL)
        self.assertIsInstance(res, dict)
        print(res)

    def tearDown(self):
        print("tests completed")


if __name__ == "__main__":
    unittest.main()
