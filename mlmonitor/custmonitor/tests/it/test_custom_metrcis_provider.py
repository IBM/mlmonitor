# SPDX-License-Identifier: Apache-2.0
import unittest
import json
import os
from unittest import TestCase
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_openscale import APIClient as WOS_APIClient
from ibm_watson_machine_learning import APIClient as WML_APIClient

# import from mlmonitor (must set MONITOR_CONFIG_PATH env variable)
from mlmonitor.src.wml import WML_URL, WML_SPACE_ID
from mlmonitor.src import API_KEY, WOS_URL
from mlmonitor.src.wml.utils import get_deployment_uid_by_name
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.wos.data_mart import get_datamart_ids
from mlmonitor.src.wos.custom_monitors import (
    get_custom_monitor_definition,
    get_custom_monitor_instance,
)
from mlmonitor.src.wos.integated_system import get_integrated_system_by_provider_name

SLOW_TESTS = int(os.getenv("RUN_SLOW_TESTS", "1"))


class TestCustomMonitor(TestCase):
    def setUp(self):

        # Based on Monitor name , find necessary ids to trigger Custom Monitor and Custom Metrics Provider
        self.monitored_deployment = "sm-cc-xgboost-2022-10-13-14-36-11-945"
        self.provider_name = "Custom_Metrics_Provider_churn"
        self.custom_monitor_name = "Custom_Metrics_Provider_churn"
        self.wml_function_deployment_name = (
            "Custom_Metrics_Provider_Deployment_churn-deploy"
        )

        self.wos_client = WOS_APIClient(authenticator=IAMAuthenticator(apikey=API_KEY))
        self.wml_client = WML_APIClient({"url": WML_URL, "apikey": API_KEY})
        self.wml_client.set.default_space(WML_SPACE_ID)

        # Deployment (Model endpoint) for which the Custom monitor is attached (via subscription)
        subscription_ids = get_subscription_id_by_deployment(
            wos_client=self.wos_client, deployment_name=self.monitored_deployment
        )
        data_marts = get_datamart_ids(wos_client=self.wos_client)
        self.data_mart_id = data_marts[0]
        if len(subscription_ids) == 1:
            self.subscription_id = subscription_ids[0]
        else:
            raise ValueError(
                f"No WOS subscription found for deployment {self.monitored_deployment}"
            )

        self.custom_metrics_provider_id = get_integrated_system_by_provider_name(
            self.wos_client, self.provider_name
        )

        if len(self.custom_metrics_provider_id) == 1:
            self.integrated_system_id = self.custom_metrics_provider_id[0]
        else:
            raise ValueError(
                f"custom provider not found with name : [{self.provider_name}]"
            )

        if not (
            existing_definition := get_custom_monitor_definition(
                wos_client=self.wos_client, monitor_name=self.custom_monitor_name
            )
        ):
            raise ValueError(
                f"custom monitor definition not found for monitor : [{self.custom_monitor_name}]"
            )

        self.monitor_definition_id = existing_definition.metadata.id
        existing_monitor_instance = get_custom_monitor_instance(
            wos_client=self.wos_client,
            data_mart_id=self.data_mart_id,
            monitor_definition_id=self.monitor_definition_id,
            subscription_id=self.subscription_id,
        )
        self.monitor_instance_id = existing_monitor_instance.metadata.id
        if not existing_monitor_instance:
            raise ValueError(
                f"custom monitor instance not found for monitor : [{self.custom_monitor_name}]"
            )
        print(
            f"- Found integrated service id [{self.integrated_system_id}] => [{existing_monitor_instance.entity.parameters.get('custom_metrics_provider_id')}]\n"
            f"- Found custom monitor definition id [{self.monitor_definition_id}] => [{existing_monitor_instance.entity.monitor_definition_id}]\n"
            f"- Found custom monitor instance id [{self.monitor_instance_id}]"
        )

        self.deployment_uid = get_deployment_uid_by_name(
            wml_client=self.wml_client,
            deployment_name=self.wml_function_deployment_name,
        )

    @unittest.skipIf(
        SLOW_TESTS == 0, "Skip over this routine test_score_metrics_provider"
    )
    def test_score_metrics_provider(self):

        monitor_instance_run_info = self.wos_client.monitor_instances.run(
            background_mode=False, monitor_instance_id=self.monitor_instance_id
        ).result
        print(monitor_instance_run_info)
        custom_monitor_run_id = monitor_instance_run_info.metadata.id

        input_data = {
            "input_data": [
                {
                    "values": {
                        "data_mart_id": self.data_mart_id,
                        "subscription_id": self.subscription_id,
                        "custom_monitor_run_id": custom_monitor_run_id,
                        "custom_monitor_id": self.monitor_definition_id,
                        "custom_monitor_instance_id": self.monitor_instance_id,
                        "custom_monitor_instance_params": {
                            "custom_metrics_provider_id": self.custom_metrics_provider_id,
                            "custom_metrics_wait_time": 300,
                        },
                    }
                }
            ]
        }

        print(f"Sending payload to WML function {self.wml_function_deployment_name}")
        job_details = self.wml_client.deployments.score(self.deployment_uid, input_data)
        print("response_payload remote\n", json.dumps(job_details, indent=3))
        self.assertIsInstance(job_details, dict)
        self.assertIn("predictions", job_details)

    def test_get_feedback_data(self):
        from metricsprovider.helpers import (
            get_access_token_cloud,
            get_feedback_dataset_id,
            get_feedback_data,
        )

        token = get_access_token_cloud(apikey=API_KEY)
        self.assertEqual(token[:2], "ey")
        feedback_dataset_id = get_feedback_dataset_id(
            access_token=token,
            data_mart_id=self.data_mart_id,
            subscription_id=self.subscription_id,
            url=WOS_URL,
        )
        self.assertIsInstance(feedback_dataset_id, str)
        response = get_feedback_data(
            token, self.data_mart_id, feedback_dataset_id, WOS_URL, 2
        )
        self.assertIn("records", response)
        self.assertListEqual(
            list(response.get("records")[0].keys()), ["annotations", "fields", "values"]
        )

    def tearDown(self):
        print("test completed")


if __name__ == "__main__":
    unittest.main()
