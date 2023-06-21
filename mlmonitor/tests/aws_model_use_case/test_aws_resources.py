# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest import TestCase

import json


class TestAWSResources(TestCase):
    def test_aws_secret_manager(self):
        from mlmonitor.src import key, secret, region, API_KEY
        from mlmonitor.use_case_churn.utils import _get_secret

        secret_name = "IBM_KEYS"
        API_KEY_NAME = "IBM_API_KEY_MLOPS"

        aws_secrets = _get_secret(
            secret_name=secret_name,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=region,
        )

        API_KEY_SECRET_MANAGER = json.loads(aws_secrets).get(API_KEY_NAME)

        self.assertEqual(API_KEY, API_KEY_SECRET_MANAGER)

    def tearDown(self):
        print("AWS Resources tests completed")


if __name__ == "__main__":
    unittest.main()
