# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest import TestCase

from mlmonitor.src import API_KEY, AUTH_ENDPOINT, ENV, USERNAME
from mlmonitor.src.factsheets.utils import FactsheetHelpers


class TestFactsheetsAuthenticate(TestCase):
    """Exercises FactsheetHelpers.authenticate() against the real IBM Cloud /
    CP4D authentication endpoint, using the credentials resolved from
    MONITOR_CONFIG_FILE (see mlmonitor/config.py). No requests mocking:
    this hits the live service the same way the rest of mlmonitor does.
    """

    @classmethod
    def setUpClass(cls):
        cls._helper = FactsheetHelpers(
            api_key=API_KEY,
            container_type="catalog",
            container_id="unused-container-id",
            model_entry_id="unused-model-entry-id",
            env=ENV,
            cpd_url=AUTH_ENDPOINT if ENV == "prem" else None,
            username=USERNAME if ENV == "prem" else None,
        )

    def test_authenticate_success(self):
        success, headers, error_msg = self._helper.authenticate()

        self.assertTrue(success)
        self.assertEqual(error_msg, "")
        self.assertIn("Authorization", headers)
        self.assertTrue(headers["Authorization"].startswith("Bearer "))
        self.assertGreater(len(headers["Authorization"]), len("Bearer "))
        self.assertEqual(headers["content-type"], "application/json")

    def test_authenticate_invalid_apikey_raises(self):
        bad_helper = FactsheetHelpers(
            api_key="definitely-not-a-valid-api-key",
            container_type="catalog",
            container_id="unused-container-id",
            model_entry_id="unused-model-entry-id",
            env=ENV,
            cpd_url=AUTH_ENDPOINT if ENV == "prem" else None,
            username=USERNAME if ENV == "prem" else None,
        )

        with self.assertRaises(RuntimeError) as ctx:
            bad_helper.authenticate()

        self.assertIn(f"env={ENV}", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
