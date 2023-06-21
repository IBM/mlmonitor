# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest import TestCase
from os import path
import os

from mlmonitor.src.model.config_aws import (
    SageMakerModelConfig,
    # SageMakerModelConfigEncoder,
)
from mlmonitor.src.model.use_case_aws import (
    SageMakerModelUseCase,
    # SageMakerModelUseCaseEncoder,
)


class TestAWS_ModelConfig_ModelUseCase(TestCase):
    def setUp(self):
        self.output_path = "./outputs"
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def test_load_model_config(self):
        mdl_cfg = SageMakerModelConfig("use_case_churn")
        self.assertEqual(mdl_cfg.train_instance, "ml.m4.xlarge")

        mdl_cfg = SageMakerModelConfig("use_case_gcr")
        self.assertEqual(mdl_cfg.train_instance, "ml.m4.xlarge")

        mdl_cfg = SageMakerModelConfig("use_case_mnist_tf")
        self.assertEqual(mdl_cfg.train_instance, "ml.c4.xlarge")

        mdl_cfg = SageMakerModelConfig("use_case_mnist_pt")
        self.assertEqual(mdl_cfg.train_instance, "ml.c4.xlarge")

    def test_load_model_use_case(self):
        CATALOG_ID = "11111111-1111-4111-1111-111111111111"
        MODEL_ENTRY_ID = "11111111-1111-4111-1111-111111111111"

        mdl_uc = SageMakerModelUseCase(
            "use_case_churn", catalog_id=CATALOG_ID, model_entry_id=MODEL_ENTRY_ID
        )
        self.assertEqual(mdl_uc.ibm_key_name, "IBM_API_KEY_MLOPS")
        self.assertEqual(mdl_uc.catalog_id, CATALOG_ID)
        self.assertEqual(mdl_uc.model_entry_id, MODEL_ENTRY_ID)

    def test_save_model_config(self):
        mdl_cfg = SageMakerModelConfig("use_case_churn")
        mdl_cfg.save_config(
            file=os.path.join(self.output_path, "/SageMakerModelConfig.json")
        )

        self.assertEqual(
            path.exists(os.path.join(self.output_path, "/SageMakerModelConfig.json")),
            True,
        )

    def test_save_model_use_case(self):
        CATALOG_ID = "11111111-1111-4111-1111-111111111111"
        MODEL_ENTRY_ID = "11111111-1111-4111-1111-111111111111"

        mdl_uc = SageMakerModelUseCase(
            "use_case_churn", catalog_id=CATALOG_ID, model_entry_id=MODEL_ENTRY_ID
        )
        mdl_uc.is_trained = True
        mdl_uc.save_use_case()

        self.assertEqual(
            path.exists(os.path.join(self.output_path, "/SageMakerModelUseCase.json")),
            True,
        )

        mdl_uc_new = SageMakerModelUseCase(
            source_dir="use_case_churn",
            file=os.path.join(self.output_path, "/SageMakerModelUseCase.json"),
        )
        self.assertEqual(mdl_uc_new.is_trained, True)

    def test_load_model_config_from_file(self):
        mdl_cfg = SageMakerModelConfig("use_case_churn")
        mdl_cfg.train_instance = "ml.m4.xs"
        mdl_cfg.save_config(
            file=os.path.join(self.output_path, "/SageMakerModelConfig.json")
        )

        mdl_cfg_new = SageMakerModelConfig(
            source_dir="use_case_churn",
            file=os.path.join(self.output_path, "/SageMakerModelConfig.json"),
        )

        self.assertEqual(mdl_cfg_new.train_instance, "ml.m4.xs")

    def tearDown(self):
        print("Model Config Cleaned up")
        if path.exists(os.path.join(self.output_path, "/SageMakerModelConfig.json")):
            os.remove(os.path.join(self.output_path, "/SageMakerModelConfig.json"))

        if path.exists(os.path.join(self.output_path, "/SageMakerModelUseCase.json")):
            os.remove(os.path.join(self.output_path, "/SageMakerModelUseCase.json"))


if __name__ == "__main__":
    unittest.main()
