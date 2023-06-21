# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest import TestCase
from os import path
import re
from typing import Optional

from mlmonitor import WMLModelUseCase

_UUID4_REGEX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_MODEL_DATA_REGEX = re.compile(r"s3://.*/([A-Za-z_0-9.-]+)/output/model.tar.gz")

_ENDPOINT_REGEX = r"-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})$"


class TestWML_Model_Onboarding(TestCase):
    @classmethod
    def setUpClass(cls):
        source_dir = os.getenv("USE_CASE")
        CATALOG_ID = (
            os.getenv("CATALOG_ID_PREM")
            if os.getenv("ENV") == "prem"
            else os.getenv("CATALOG_ID_SAAS")
        )
        MODEL_ENTRY_ID = (
            os.getenv("MODEL_ENTRY_ID_PREM")
            if os.getenv("ENV") == "prem"
            else os.getenv("MODEL_ENTRY_ID_SAAS")
        )

        cls._wml_use_case = WMLModelUseCase(
            source_dir=source_dir, catalog_id=CATALOG_ID, model_entry_id=MODEL_ENTRY_ID
        )

    @classmethod
    def tearDownClass(cls):
        print(f"Completed {os.getenv('USE_CASE')} test case")

    def test_onboard_model(self):
        self._assert_states(False, False, False, False)
        self._wml_use_case.train()

        self.assertRegexpMatches(self._wml_use_case.model_data, _MODEL_DATA_REGEX)
        self.model_endpoint = self._wml_use_case.model_endpoint

        self._assert_states(True, False, False, False)
        sm_loaded_uc = self._validate_save_model_uc(
            f"./wml_use_case_{os.getenv('USE_CASE')}_trained.json"
        )
        self._assert_states(True, False, False, False, sm_loaded_uc)
        self._wml_use_case.deploy()

        resp = self._wml_use_case.score_model()

        self.assertSetEqual(set(resp.get("fields")), {"score", "predicted_label"})

        self._assert_states(True, True, False, False)
        self.assertRegexpMatches(
            self._wml_use_case.model_endpoint,
            re.compile(
                self._wml_use_case._model_config.base_job_name + _ENDPOINT_REGEX
            ),
        )

        self._wml_use_case.govern()
        self._assert_states(True, True, True, False)
        self._wml_use_case.monitor()
        self._assert_states(True, True, True, True)
        self.assertRegexpMatches(self._wml_use_case.subscription_id, _UUID4_REGEX)

        if self._wml_use_case._model_config.quality_monitor_enabled:
            res = self._wml_use_case.configure_quality_monitor()
            self.assertRegexpMatches(res.get("quality"), _UUID4_REGEX)

        if self._wml_use_case._model_config.fairness_monitor_enabled:
            res = self._wml_use_case.configure_fairness_monitor()
            self.assertRegexpMatches(res.get("fairness"), _UUID4_REGEX)

        if self._wml_use_case._model_config.explain_monitor_enabled:
            res = self._wml_use_case.configure_explainability_monitor()
            self.assertRegexpMatches(res.get("explainability"), _UUID4_REGEX)

        if self._wml_use_case._model_config.drift_monitor_enabled:
            res = self._wml_use_case.configure_drift_monitor()
            self.assertRegexpMatches(res.get("drift"), _UUID4_REGEX)

        self._wml_use_case.log_payload()
        self._wml_use_case.log_feedback()

        self._wml_use_case.evaluate_model()
        sm_loaded_uc = self._validate_save_model_uc(
            f"./wml_use_case_{os.getenv('USE_CASE')}_completed.json"
        )
        self._assert_states(True, True, True, True, sm_loaded_uc)

        res = self._wml_use_case.cleanup()

        self._assert_states(True, False, False, False)

    def _validate_save_model_uc(self, file: str) -> WMLModelUseCase:
        """
        validate the save_use_case method of the SageMakerModelUseCase class.

        :param file: str: path to save SageMakerModelUseCase object
        :return: A SageMakerModelUseCase object loaded

        """
        self._wml_use_case.save_use_case(file=file)

        self.assertEqual(path.exists(file), True)

        return WMLModelUseCase(source_dir=os.getenv("USE_CASE"), file=file)

    def _assert_states(
        self,
        trained: bool,
        deployed: bool,
        governed: bool,
        monitored: bool,
        use_case: Optional[WMLModelUseCase] = None,
    ):
        """
        validate states WMLModelUseCase object.

        :param trained: bool: current trained status
        :param deployed: bool: current trained status
        :param governed: bool: current trained status
        :param monitored: bool: current trained status
        :param use_case: Optional[WMLModelUseCase]: model use case instance
        :return: None
        """
        if not use_case:
            use_case = self._wml_use_case

        self.assertEqual(use_case.is_trained, trained)
        self.assertEqual(use_case.is_deployed, deployed)
        self.assertEqual(use_case.is_governed, governed)
        self.assertEqual(use_case.is_monitored, monitored)


if __name__ == "__main__":
    unittest.main()
