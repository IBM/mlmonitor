# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Union
import json
import os
import re

from mlmonitor.src.model.config import ModelConfig
from mlmonitor.src import logger
from mlmonitor.src.aws import SUPPORTED_SAGEMAKER_ESTIMATORS
from mlmonitor.exceptions import ModelSignatureError


class SageMakerModelConfigEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, SageMakerModelConfig):
            return obj.json_object()
        return super().default(obj)


class SageMakerModelConfig(ModelConfig):
    """SageMakerModelConfig inherits from ModelConfig abstract class containing all Model specific attributes and methods to perform the following actions :

    - training in AWS Sagemaker (specific to this class)
    - Deployment in AWS Sagemaker (specific to this class)
    - Monitoring in Watson OpenScale
    - Governance in AI Factsheets

    """

    def __init__(self, source_dir: str, file: str = None):
        super().__init__(source_dir=source_dir, file=file)

        self.load_aws_config(file=self.config_file)

    @property
    def base_job_name(self) -> str:
        """ ""Model Use Case base job_name property for Sagemaker"""
        return self._base_job_name

    @base_job_name.setter
    def base_job_name(self, value: str):
        """
        Set the base job_name to use in SageMaker jobs
        :param value:
        :return:
        """
        self._base_job_name = value

    @property
    def train_script(self) -> str:
        """train_script property corresponds to training .py executed for model training"""
        return self._train_script

    @train_script.setter
    def train_script(self, value: str):
        """
        Set the train_script to be used (must be a valid executable python file)
        :param value:
        :return:
        """
        if self.valid_pyfile(filename=value):
            self._train_script = value

    @property
    def train_py_version(self) -> str:
        """train_py_version Config property"""
        return self._train_py_version

    @train_py_version.setter
    def train_py_version(self, value: str):
        """
        Set the train_py_version to use : training python container version to be used
        :param value:
        :return:
        """
        self._train_py_version = value

    @property
    def train_framework(self) -> str:
        """train_framework Config property"""
        return self._train_framework

    @train_framework.setter
    def train_framework(self, value: str):
        """
        Set the train_framework to be used (container type for training job)
        :param value:
        :return:
        """
        if value.lower() in SUPPORTED_SAGEMAKER_ESTIMATORS:
            self._train_framework = value.lower()
        else:
            raise ModelSignatureError.invalid_signature_value(
                supported_values=SUPPORTED_SAGEMAKER_ESTIMATORS,
                passed_value=value,
                field_name="train_framework",
                section_name="aws_runtime",
            )

    @property
    def train_framework_version(self) -> str:
        """train_framework_version Config property"""

        return self._train_framework_version

    @train_framework_version.setter
    def train_framework_version(self, value: str):
        """
        Set the train_framework_version to use (container framework version for training job)
        :param value:
        :return:
        """
        self._train_framework_version = value

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
        assert value in {"sklearn", "xgboost", "pytorch", "tensorflow"}
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
    def train_instance(self) -> str:
        """train_instance Config property"""
        return self._train_instance

    @train_instance.setter
    def train_instance(self, value: str):
        """
        Set the train_instance type to use by training job
        :param value:
        :return:
        """
        if not re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", value):
            logger.error(
                "Invalid SageMaker instance type for training "
                "https://aws.amazon.com/sagemaker/pricing/instance-types"
            )
        else:
            self._train_instance = value

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
        if not re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", value):
            logger.error(
                "Invalid SageMaker instance type for inference "
                "https://aws.amazon.com/sagemaker/pricing/instance-types"
            )
        else:
            self._inference_instance = value

    @property
    def prefix(self) -> str:
        """prefix Config property for training job name assignment"""
        return self._prefix

    @prefix.setter
    def prefix(self, value: str):
        """
        Set the prefix to use for training job name assignment
        :param value: prefix to use for training
        :return:
        """
        self._prefix = value

    @property
    def serializer(self) -> str:
        """serializer Config property for serializer type in inference instance"""
        return self._serializer

    @serializer.setter
    def serializer(self, value: str):
        """
        Set the serializer type to use in inference instance
        :param value:
        :return:
        """
        self._serializer = value

    @property
    def deserializer(self) -> str:
        """deserializer Config property"""
        return self._deserializer

    @deserializer.setter
    def deserializer(self, value: str):
        """
        Set the deserializer to use in inference instance
        :param value:
        :return:
        """
        self._deserializer = value

    def save_config(self, file: str = None):
        """
        saves the SageMakerModelConfig class to a JSON file.
        accepts an optional argument, file, which is the path to  the JSON file that will be created.
        If no value  is passed in for the optional argument, then save_config creates a new JSON file
        named SageMakerModelConfig.json in the current working directory.

        :param self: Access the class variables
        :param file:str=None: Specify a file path to save the sagemakermodelconfig class
        :return: The json string of the sagemakermodelconfig class
        """
        if not file:
            file = os.path.join("./", "SageMakerModelConfig.json")
        logger.info(f"SageMakerModelConfig class saved to {file}")
        with open(file, "w") as f:
            json.dump(obj=self, fp=f, cls=SageMakerModelConfigEncoder, indent=4)

    def json_object(self):
        """
        helper function that returns the json object of the SageMakerModelConfig class.
        takes in an instance of a ModelConfig class and returns its json object.

        :param self: Access the class instance inside of a method
        :return: A dictionary that combines the contents of the base_config and sm_config dictionaries
        """
        SageMakerModelConfigObject = self.sm_config_json()
        ModelConfigObject = self.base_config_json()
        ModelConfigObject["__class__"] = "SageMakerModelConfig"
        return {**ModelConfigObject, **SageMakerModelConfigObject}

    def load_aws_config(self, file: Union[str, Dict] = None):
        """
        Method Used to infer class parameters for a new SageMakerModelConfig instance based on JSON dictionary
        generated by json_object`

        :param self: Access the attributes and methods of the class in python
        :param file:Union[str,Dict]: Specify the path to a file containing the sagemakermodelconfig serialized Object in JSON
        :return:
        """

        if isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
                logger.info(f"SageMakerModelConfig class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logger.info("SageMakerModelConfig class loaded from dict")
        else:
            raise ValueError("should be a file path or dictionary")

        # training
        self._runtime = loaded_dict.get("aws_runtime")

        self.base_job_name = self._runtime.get("job_name")
        self.train_script = self._runtime.get("train_script")
        self.train_py_version = self._runtime.get("train_py_version")
        self.train_framework = self._runtime.get("train_framework")
        self.train_framework_version = self._runtime.get("train_framework_version")
        self.train_instance = self._runtime.get("train_instance")

        # inference
        self.inference_script = self._runtime.get("inference_script")
        self.inference_py_version = self._runtime.get("inference_py_version")
        self.inference_framework = self._runtime.get("inference_framework")
        self.inference_framework_version = self._runtime.get(
            "inference_framework_version"
        )
        self.inference_instance = self._runtime.get("inference_instance")

        self.prefix = self._runtime.get("prefix")
        self.serializer = self._runtime.get("serializer")
        self.deserializer = self._runtime.get("deserializer")

    def sm_config_json(self) -> Dict:
        return {
            "aws_runtime": {
                "train_script": self.train_script,
                "inference_script": self.inference_script,
                "train_framework": self.train_framework,
                "train_framework_version": self.train_framework_version,
                "train_py_version": self.train_py_version,
                "inference_framework": self.inference_framework,
                "inference_framework_version": self.inference_framework_version,
                "train_instance": self.train_instance,
                "inference_instance": self.inference_instance,
                "inference_py_version": self.inference_py_version,
                "job_name": self.base_job_name,
                "prefix": self.prefix,
                "serializer": self.serializer,
                "deserializer": self.deserializer,
            },
        }
