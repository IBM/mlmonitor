# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Union, Optional
import json
import os
import importlib

from mlmonitor.exceptions import ModelSignatureError
from mlmonitor.src.model.config import ModelConfig
from mlmonitor.src import logger

try:
    from azureml.core.compute import ComputeTarget
    from azureml.core.compute_target import ComputeTargetException
    from mlmonitor.src.azure import AZ_WORKSPACE, SUPPORTED_AZURE_COMPUTE
except ModuleNotFoundError:
    logger.warning("run pip install mlmonitor[azure] to use AzureModelUseCase")


class AzureModelConfigEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, AzureModelConfig):
            return obj.json_object()
        return super().default(obj)


class AzureModelConfig(ModelConfig):
    """AzureModelConfig inherits from ModelConfig abstract class containing all Model specific attributes and methods to perform the following actions :

    - training in  Azure (specific to this class)
    - Deployment in Azure (specific to this class)
    - Monitoring in Watson OpenScale
    - Governance in AI FactSheets

    """

    def __init__(self, source_dir: str, file: str = None):
        super().__init__(source_dir=source_dir, file=file)

        self.load_azure_config(file=self.config_file)

    @property
    def train_script(self) -> str:
        """train_script property corresponds to training .py executed for model training in jobs"""
        return self._train_script

    @train_script.setter
    def train_script(self, value: str):
        """
        Set the train_script to be used (must be a valid executable python file) in azure jobs
        :param value:
        :return:
        """
        if self.valid_pyfile(filename=value):
            self._train_script = value

    @property
    def train_module(self) -> str:
        """train_module property corresponds to training model used for model training on local environment"""
        return self._train_module

    @train_module.setter
    def train_module(self, value: str):
        """
        Set the train_module to be used to train model on local environment
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
    def inference_compute(self) -> str:
        """inference_compute property corresponds to the inference compute in Azure"""
        return self._inference_compute

    @inference_compute.setter
    def inference_compute(self, value: str):
        """
        Set inference_compute to be used (must be aks or aci)
        AksWebservice https://learn.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.akswebservice?view=azure-ml-py
        AciWebservice https://learn.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.aci.aciwebservice?view=azure-ml-py
        :param value:
        :return:
        """
        if value.lower() in SUPPORTED_AZURE_COMPUTE:
            self._inference_compute = value.lower()
        else:
            raise ModelSignatureError.invalid_signature_value(
                "inference_compute", value, SUPPORTED_AZURE_COMPUTE, "azure_runtime"
            )

    @property
    def aks_cluster_name(self) -> Optional[str]:
        """aks_cluster_name Config property"""
        return self._aks_cluster_name

    @aks_cluster_name.setter
    def aks_cluster_name(self, value: Optional[str] = None):
        """
        Set the aks_cluster_name to use : aks_cluster_name should exist in  Azure environment
        :param value:
        :return:
        """
        if value and self.inference_compute == "aks":
            try:
                ComputeTarget(workspace=AZ_WORKSPACE, name=value)
                logger.debug(
                    f"Found existing cluster {value} in workspace {AZ_WORKSPACE.name}"
                )

            except ComputeTargetException as e:
                raise ModelSignatureError(
                    f"aks_cluster_name property cannot be set [{e.message}]"
                ) from e
        self._aks_cluster_name = value

    @property
    def cpu_cores(self) -> Union[float, int]:
        """cpu_cores Config property"""
        return self._cpu_cores

    @cpu_cores.setter
    def cpu_cores(self, value: Union[float, int]):
        """
        Set the cpu_cores to assign in Azure deployment
        :param value:
        :return:
        """
        if isinstance(value, (int, float)):
            self._cpu_cores = min(value, 4)
        else:
            raise ModelSignatureError("cpu_cores should be float")

    @property
    def memory_gb(self) -> Union[float, int]:
        """memory_gb Config property"""
        return self._memory_gb

    @memory_gb.setter
    def memory_gb(self, value: Union[float, int]):
        """
        Set the memory_gb to assign in Azure deployment
        :param value:
        :return:
        """
        if isinstance(value, (int, float)):
            self._memory_gb = min(value, 2)
        else:
            raise ModelSignatureError("memory_gb should be float")

    @property
    def auth_enabled(self) -> bool:
        """auth_enabled Config property"""
        return self._auth_enabled

    @auth_enabled.setter
    def auth_enabled(self, value: Optional[bool] = True):
        """
        Set the auth_enabled for Azure deployment
        :param value:
        :return:
        """
        if isinstance(value, bool):
            self._auth_enabled = value
        else:
            raise ModelSignatureError("auth_enabled should be boolean type")

    @property
    def tags(self) -> dict:
        """tags Config property"""
        return self._tags

    @tags.setter
    def tags(self, value: dict):
        """
        Set the tags for Azure deployment
        :param value:
        :return:
        """
        self._tags = value

    @property
    def description(self) -> str:
        """description Config property"""
        return self._description

    @description.setter
    def description(self, value: str):
        """
        Set the description for Azure deployment
        :param value:
        :return:
        """
        self._description = value

    @property
    def conda_packages(self) -> list:
        """conda_packages Config property"""
        return self._conda_packages

    @conda_packages.setter
    def conda_packages(self, value: list):
        """
        Set the conda_packages to use : conda_packages to be installed in azure inference container
        :param value:
        :return:
        """
        self._conda_packages = value

    @property
    def pip_packages(self) -> list:
        """pip_packages Config property"""
        return self._pip_packages

    @pip_packages.setter
    def pip_packages(self, value: list):
        """
        Set the pip_packages to use : pip_packages to be installed in azure inference and training container
        :param value:
        :return:
        """
        self._pip_packages = value

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

    def save_config(self, file: str = None):
        """
        saves the AzureModelConfig class to a JSON file.
        accepts an optional argument, file, which is the path to  the JSON file that will be created.
        If no value  is passed in for the optional argument, then save_config creates a new JSON file
        named AzureModelConfig.json in the current working directory.

        :param self: Access the class variable
        :param file:str=None: Specify a file path to save the Azuremodelconfig class
        :return: The json string of the Azuremodelconfig class
        """
        if not file:
            file = os.path.join("./", "AzureModelConfig.json")
        logger.info(f"AzureModelConfig class saved to {file}")
        with open(file, "w") as f:
            json.dump(obj=self, fp=f, cls=AzureModelConfigEncoder, indent=4)

    def json_object(self):
        """
        helper function that returns the json object of the AzureModelConfig class.
        takes in an instance of a ModelConfig class and returns its json object.

        :param self: Access the class instance inside a method
        :return: A dictionary that combines the contents of the base_config and sm_config dictionaries
        """
        AzureModelConfigObject = self.azure_config_json()
        ModelConfigObject = self.base_config_json()
        ModelConfigObject["__class__"] = "AzureModelConfig"
        return {**ModelConfigObject, **AzureModelConfigObject}

    def load_azure_config(self, file: Union[str, Dict] = None):
        """
        Method Used to infer class parameters for a new AzureModelConfig instance based on JSON dictionary
        generated by json_object`

        :param self: Access the attributes and methods of the class in python
        :param file:Union[str,Dict]: Specify the path to a file containing the Azuremodelconfig serialized Object in JSON
        :return:
        """

        if isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
                logger.info(f"AzureModelConfig class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logger.info("AzureModelConfig class loaded from dict")
        else:
            raise ValueError("should be a file path or dictionary")

        # training
        self._runtime = loaded_dict.get("azure_runtime")

        # inference
        self.train_script = self._runtime.get("train_script")
        self.train_py_version = self._runtime.get("train_py_version")
        self.inference_script = self._runtime.get("inference_script")
        self.inference_compute = self._runtime.get("inference_compute")
        self.aks_cluster_name = self._runtime.get("aks_cluster_name")
        self.cpu_cores = self._runtime.get("cpu_cores")
        self.memory_gb = self._runtime.get("memory_gb")
        self.auth_enabled = self._runtime.get("auth_enabled")
        self.description = self._runtime.get("description")
        self.tags = self._runtime.get("tags")
        self.conda_packages = self._runtime.get("conda_packages")
        self.pip_packages = self._runtime.get("pip_packages")
        self.inference_py_version = self._runtime.get("inference_py_version")
        self.train_module = self._runtime.get("train_module")
        self.train_method = self._runtime.get("train_method")

        # For Azure ML models prediction_field should be "Scored Labels"
        # probability fields should be "Scored Probabilities"
        # https://www.ibm.com/docs/en/cloud-paks/cp-data/4.6.x?topic=models-microsoft-azure-ml-service-frameworks
        # return fields are ["Scored Labels", "Scored Probabilities"]
        self.prediction_field = "Scored Labels"
        self.probability_fields = ["Scored Probabilities"]

    def azure_config_json(self) -> Dict:
        return {
            "azure_runtime": {
                "train_script": self.train_script,
                "train_py_version": self.train_py_version,
                "inference_script": self.inference_script,
                "inference_compute": self.inference_compute,
                "aks_cluster_name": self.aks_cluster_name,
                "cpu_cores": self.cpu_cores,
                "memory_gb": self.memory_gb,
                "auth_enabled": self.auth_enabled,
                "description": self.description,
                "tags": self.tags,
                "conda_packages": self.conda_packages,
                "pip_packages": self.pip_packages,
                "inference_py_version": self.inference_py_version,
                "train_module": self.train_module,
                "train_method": self.train_method,
            },
        }
