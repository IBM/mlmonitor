# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest import TestCase

from mlmonitor.src import API_KEY, AUTH_ENDPOINT, ENV, USERNAME
from mlmonitor.src.factsheets.utils import FactsheetHelpers


class TestFactsheetsGetModelEntryContent(TestCase):
    """Exercises FactsheetHelpers.get_model_entry_content() against the real
    AI Factsheets REST API (no requests mocking). Requires an existing model
    use case in the target catalog:

      CATALOG_ID_SAAS / MODEL_ENTRY_ID_SAAS  (env=saas)
      CATALOG_ID_PREM / MODEL_ENTRY_ID_PREM  (env=prem)
    """

    @classmethod
    def setUpClass(cls):
        catalog_id = (
            os.getenv("CATALOG_ID_PREM")
            if ENV == "prem"
            else os.getenv("CATALOG_ID_SAAS")
        )
        model_entry_id = (
            os.getenv("MODEL_ENTRY_ID_PREM")
            if ENV == "prem"
            else os.getenv("MODEL_ENTRY_ID_SAAS")
        )

        if not catalog_id or not model_entry_id:
            raise unittest.SkipTest(
                "CATALOG_ID_SAAS/MODEL_ENTRY_ID_SAAS (or _PREM) env vars must be "
                "set to an existing model use case to run this test"
            )

        cls._helper = FactsheetHelpers(
            api_key=API_KEY,
            container_type="catalog",
            container_id=catalog_id,
            model_entry_id=model_entry_id,
            env=ENV,
            cpd_url=AUTH_ENDPOINT if ENV == "prem" else None,
            username=USERNAME if ENV == "prem" else None,
        )

    def test_get_model_entry_content(self):
        content = self._helper.get_model_entry_content()

        self.assertIsInstance(content, dict)
        self.assertNotIn("errors", content, content.get("errors"))
        self.assertIn("resources", content)
        resources = content["resources"]
        self.assertIsInstance(resources, list)
        self.assertGreater(len(resources), 0)
        for resource in resources:
            self.assertEqual(resource.get("asset_id"), self._helper.model_entry_id)
        self.assertTrue(
            any("model_entry" in resource for resource in resources),
            "expected at least one resource with a 'model_entry' attribute",
        )


if __name__ == "__main__":
    unittest.main()
