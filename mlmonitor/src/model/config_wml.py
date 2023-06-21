# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Union
import json
import os
import importlib
from mlmonitor.src.model.config import ModelConfig
from mlmonitor.src import logger
from mlmonitor.src.wml import SUPPORTED_WML_RUNTIMES
from mlmonitor.exceptions import ModelSignatureError


class WMLModelconfigEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, WMLModelconfig):
            return obj.json_object()
        return super().default(obj)


class WMLModelconfig(ModelConfig):
    """WMLModelconfig inherits from ModelConfig abstract class containing all Model specific attributes and methods to perform the following actions :

    - training in local environment (specific to this class)
    - Deployment in WML deployment space (specific to this class)
    - Monitoring in Watson OpenScale
    - Governance in AI Factsheets

    """

    def __init__(self, source_dir: str, file: str = None):
        super().__init__(source_dir=source_dir, file=file)

        self.load_wml_config(file=self.config_file)

    @property
    def train_module(self) -> str:
        """train_module property corresponds to training model used for model training"""
        return self._train_module

    @train_module.setter
    def train_module(self, value: str):
        """
        Set the train_module to be used to train model locally
        :param value:
        :return:
        """
        try:
            importlib.import_module(f"mlmonitor.{self._source_dir}.{value}")
            logger.info(f"{self._source_dir}.{value} is a valid python module")
        except ModuleNotFoundError as e:
            logger.error(f" {value} should be a valid python module {e}")
        self._train_module = value

    @property
    def train_method(self) -> str:
        """train_method corresponds to training function used for model training"""
        return self._train_method

    @train_method.setter
    def train_method(self, value: str):
        """
        Set the train_method to be used to train model locally
        :param value:
        :return:
        """
        self._train_method = value

    @property
    def inference_script(self) -> str:
        """inference_script Config property"""
        return self._inference_script

    @inference_script.setter
    def inference_script(self, value: str):
        """
        Set the inference_script to be used for inference (must be a valid executable python file)
        :param value:
        :return:
        """
        if self.valid_pyfile(filename=value):
            self._inference_script = value

    @property
    def inference_py_version(self) -> str:
        """inference_py_version Config property"""
        return self._inference_py_version

    @inference_py_version.setter
    def inference_py_version(self, value: str):
        """
        Set the inference_py_version to use (container framework version for training job)
        :param value:
        :return:
        """
        self._inference_py_version = value

    @property
    def inference_framework(self) -> str:
        """inference_framework Config property"""
        return self._inference_framework

    @inference_framework.setter
    def inference_framework(self, value: str):
        """
        Set the inference_framework to use (container framework for model endpoints)
        :param value:
        :return:
        """
        assert value in {"scikit-learn", "tensorflow", "xgboost"}
        self._inference_framework = value

    @property
    def inference_framework_version(self) -> str:
        """inference_framework_version Config property"""
        return self._inference_framework_version

    @inference_framework_version.setter
    def inference_framework_version(self, value: str):
        """
        Set the inference_framework to use  (container framework for endpoint)
        :param value:
        :return:
        """
        self._inference_framework_version = value

    @property
    def inference_instance(self) -> str:
        """inference_instance Config property"""
        return self._inference_instance

    @inference_instance.setter
    def inference_instance(self, value: str):
        """
        Set the inference_instance to use in Sagemaker endpoint deployment
        :param value:
        :return:
        """
        if value in SUPPORTED_WML_RUNTIMES:
            self._inference_instance = value

        else:
            raise ModelSignatureError.invalid_signature_value(
                supported_values=SUPPORTED_WML_RUNTIMES,
                passed_value=value,
                field_name="inference_instance",
                section_name="wml_runtime",
            )

    def save_config(self, file: str = None):
        """
        saves the WMLModelConfig class to a JSON file.
        accepts an optional argument, file, which is the path to  the JSON file that will be created.
        If no value  is passed in for the optional argument, then save_config creates a new JSON file
        named WMLModelconfig.json in the current working directory.

        :param self: Access the class variables
        :param file:str=None: Specify a file path to save the WMLModelconfig class
        :return: The json string of the WMLModelconfig class
        """
        if not file:
            file = os.path.join("./", "WMLModelconfig.json")
        logger.info(f"WMLModelconfig class saved to {file}")
        with open(file, "w") as f:
            json.dump(obj=self, fp=f, cls=WMLModelconfigEncoder, indent=4)

    def json_object(self):
        """
        helper function that returns the json object of the WMLModelconfig class.
        takes in an instance of a ModelConfig class and returns its json object.

        :param self: Access the class instance inside a method
        :return: A dictionary that combines the contents of the base_config and sm_config dictionaries
        """
        WMLModelconfigObject = self.wml_config_json()
        ModelConfigObject = self.base_config_json()
        ModelConfigObject["__class__"] = "WMLModelconfig"
        return {**ModelConfigObject, **WMLModelconfigObject}

    def load_wml_config(self, file: Union[str, Dict] = None):
        """
        Method Used to infer class parameters for a new WMLModelconfig instance based on JSON dictionary
        generated by json_object`

        :param self: Access the attributes and methods of the class in python
        :param file:Union[str,Dict]: Specify the path to a file containing the WMLModelconfig serialized Object in JSON
        :return:
        """

        if isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
                logger.info(f"WMLModelconfig class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logger.info("WMLModelconfig class loaded from dict")
        else:
            raise ValueError("should be a file path or dictionary")

        # training
        self._runtime = loaded_dict.get("wml_runtime")

        self.train_module = self._runtime.get("train_module")
        self.train_method = self._runtime.get("train_method")

        # inference
        self.inference_instance = self._runtime.get("inference_instance")
        self.inference_script = self._runtime.get("inference_script")
        self.inference_py_version = self._runtime.get("inference_py_version")
        self.inference_framework = self._runtime.get("inference_framework")
        self.inference_framework_version = self._runtime.get(
            "inference_framework_version"
        )

        # For WML models prediction_field should be "prediction"
        # probability fields should be "probability"
        # return fields are ['prediction', 'prediction_classes', 'probability']
        self.prediction_field = "prediction"
        self.probability_fields = ["probability"]

    def wml_config_json(self) -> Dict:
        return {
            "wml_runtime": {
                "train_module": self.train_module,
                "train_method": self.train_method,
                "inference_script": self.inference_script,
                "inference_framework": self.inference_framework,
                "inference_framework_version": self.inference_framework_version,
                "inference_instance": self.inference_instance,
                "inference_py_version": self.inference_py_version,
            }
        }
