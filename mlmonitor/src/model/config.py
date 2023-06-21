# SPDX-License-Identifier: Apache-2.0
import os
import json
import pandas as pd
import random
import importlib
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple

from mlmonitor.src import logger, DATA_ROOT
from mlmonitor.src.utils.validation import is_csv


class ModelConfig(ABC):
    """
    ModelConfig define a standard interface for Model Configuration agnostic of model serving environment
    Model Serving specific Config Classes (e.g SageMakerModelConfig ) must inherit from this parent Class.
    """

    def __init__(self, source_dir: str, file: str = None):

        self.source_dir = source_dir
        # calls `module` setter to load `source_dir` python module
        self.module = source_dir
        # calls `config_file` to set config file from which Object should be loaded
        self.config_file = file
        self.load_config(file=self.config_file)

    @property
    def data_dir(self):
        """data_dir Config property"""
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value: str):
        """
        Set the data_dir to use to instantiate Config Class
        :param value:
        :return:
        """
        self._data_dir = value

    @property
    def validation_data(self) -> str:
        """validation_data Config property"""
        return self._validation_data

    @validation_data.setter
    def validation_data(self, value: str):
        """
        Set the valid_dataset to use
        :param value:
        :return:
        """
        self._validation_data = value

    @property
    def training_data(self) -> str:
        """training_data Config property"""
        return self._training_data

    @training_data.setter
    def training_data(self, value: str):
        """
        Set the train_dataset to use
        :param value:
        :return:
        """
        self._training_data = value

    @property
    def test_data(self) -> str:
        """Config property"""
        return self._test_data

    @test_data.setter
    def test_data(self, value: str):
        """
        Set the test_data to use
        :param value:
        :return:
        """
        self._test_data = value

    @property
    def fetch_data_function(self) -> str:
        """fetch_data_function Config property"""
        return self._fetch_data_function

    @fetch_data_function.setter
    def fetch_data_function(self, value: str):
        """
        Set the fetch_data_function to use - used for unstructured data types
        :param value:
        :return:
        """
        self._fetch_data_function = value

    @property
    def fetch_data_module(self) -> str:
        """fetch_data_module Config property"""
        return self._fetch_data_module

    @fetch_data_module.setter
    def fetch_data_module(self, value: str):
        """
        Set the fetch_data_module to use - to fetch training data
        :param value:
        :return:
        """
        self._fetch_data_module = value

    @property
    def feature_columns(self) -> List:
        """feature_columns Config property"""
        return self._feature_columns

    @feature_columns.setter
    def feature_columns(self, value: List):
        """
        Set the feature_columns to use
        :param value:
        :return:
        """
        self._feature_columns = value

    @property
    def problem_type(self) -> str:
        """problem_type Config property"""
        return self._problem_type

    @problem_type.setter
    def problem_type(self, value: str):
        """
        Set the problem_type to use
        :param value:
        :return:
        """
        self._problem_type = value

    @property
    def data_type(self) -> str:
        """data_type Config property"""
        return self._data_type

    @data_type.setter
    def data_type(self, value: str):
        """
        Set the data_type to use
        :param value:
        :return:
        """
        self._data_type = value

    @property
    def description(self) -> str:
        """description Config property"""
        return self._description

    @description.setter
    def description(self, value: str):
        """
        Set the data_type to use
        :param value:
        :return:
        """
        self._description = value

    @property
    def class_label(self) -> str:
        """class_label Config property"""
        return self._class_label

    @class_label.setter
    def class_label(self, value: str):
        """
        Set the data_type to use
        :param value:
        :return:
        """
        self._class_label = value

    @property
    def prediction_field(self) -> str:
        """prediction_field Config property"""
        return self._prediction_field

    @prediction_field.setter
    def prediction_field(self, value: str):
        """
        Set the prediction_field to use
        :param value:
        :return:
        """
        self._prediction_field = value

    @property
    def probability_fields(self) -> List:
        """probability_fields Config property"""
        return self._probability_fields

    @probability_fields.setter
    def probability_fields(self, value: List):
        """
        Set the probability_fields to use
        :param value:
        :return:
        """
        self._probability_fields = value

    @property
    def categorical_columns(self) -> List:
        """categorical_columns Config property"""
        return self._categorical_columns

    @categorical_columns.setter
    def categorical_columns(self, value: List):
        """
        Set the categorical_columns to use
        :param value:
        :return:
        """
        self._categorical_columns = value

    @property
    def hyperparameters(self) -> Dict:
        """hyperparameters Config property"""
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: Dict):
        """
        Set the hyperparameters to use
        :param value:
        :return:
        """
        self._hyperparameters = value

    @property
    def quality_monitor_enabled(self) -> bool:
        """quality_monitor_enabled Config property"""
        return self._quality_monitor_enabled

    @quality_monitor_enabled.setter
    def quality_monitor_enabled(self, value: bool):
        """
        Set the quality_monitor_enabled to use
        :param value:
        :return:
        """
        self._quality_monitor_enabled = value

    @property
    def quality_monitor_parameters(self) -> Dict:
        """quality_monitor_parameters Config property"""
        return self._quality_monitor_parameters

    @quality_monitor_parameters.setter
    def quality_monitor_parameters(self, value: Dict):
        """
        Set the quality_monitor_params to use
        :param value:
        :return:
        """
        self._quality_monitor_parameters = value

    @property
    def quality_monitor_thresholds(self) -> Dict:
        """quality_monitor_thresholds Config property"""
        return self._quality_monitor_thresholds

    @quality_monitor_thresholds.setter
    def quality_monitor_thresholds(self, value: Dict):
        """
        Set the quality_monitor_thresholds to use
        :param value:
        :return:
        """
        self._quality_monitor_thresholds = value

    @property
    def fairness_monitor_enabled(self) -> bool:
        """fairness_monitor_enabled Config property"""
        return self._fairness_monitor_enabled

    @fairness_monitor_enabled.setter
    def fairness_monitor_enabled(self, value: bool):
        """
        Set the fairness_monitor_enabled to use
        :param value:
        :return:
        """
        self._fairness_monitor_enabled = value

    @property
    def fairness_monitor_parameters(self) -> Dict:
        """fairness_monitor_parameters Config property"""
        return self._fairness_monitor_parameters

    @fairness_monitor_parameters.setter
    def fairness_monitor_parameters(self, value: Dict):
        """
        Set the fairness_monitor_parameters to use
        :param value:
        :return:
        """
        self._fairness_monitor_parameters = value

    @property
    def drift_monitor_enabled(self) -> bool:
        """drift_monitor_enabled Config property"""
        return self._drift_monitor_enabled

    @drift_monitor_enabled.setter
    def drift_monitor_enabled(self, value: bool):
        """
        Set the drift_monitor_enabled to use
        :param value:
        :return:
        """
        self._drift_monitor_enabled = value

    @property
    def drift_monitor_parameters(self) -> Dict:
        """drift_monitor_parameters Config property"""
        return self._drift_monitor_parameters

    @drift_monitor_parameters.setter
    def drift_monitor_parameters(self, value: Dict):
        """
        Set the fairness_monitor_parameters to use
        :param value:
        :return:
        """
        self._drift_monitor_parameters = value

    @property
    def drift_learn_constraints(self) -> Dict:
        """drift_learn_constraints Config property"""
        return self._drift_learn_constraints

    @drift_learn_constraints.setter
    def drift_learn_constraints(self, value: Dict):
        """
        Set the drift_learn_constraints to use
        :param value:
        :return:
        """
        self._drift_learn_constraints = value

    @property
    def explain_monitor_enabled(self) -> bool:
        """explain_monitor_enabled Config property"""
        return self._explain_monitor_enabled

    @explain_monitor_enabled.setter
    def explain_monitor_enabled(self, value: bool):
        """
        Set the explain_monitor_enabled to use
        :param value:
        :return:
        """
        self._explain_monitor_enabled = value

    @property
    def mrm_monitor_enabled(self) -> bool:
        """mrm_monitor_enabled Config property"""
        return self._mrm_monitor_enabled

    @mrm_monitor_enabled.setter
    def mrm_monitor_enabled(self, value: bool):
        """
        Set the mrm_monitor_enabled to use
        :param value:
        :return:
        """
        self._mrm_monitor_enabled = value

    @property
    def custom_monitor_enabled(self) -> bool:
        """custom_monitor_enabled Config property"""
        return self._custom_monitor_enabled

    @custom_monitor_enabled.setter
    def custom_monitor_enabled(self, value: bool):
        """
        Set the custom_monitor_enabled to use
        :param value:
        :return:
        """
        self._custom_monitor_enabled = value

    @property
    def custom_monitor_names(self) -> str:
        """custom_monitor_names Config property"""
        return self._custom_monitor_names

    @custom_monitor_names.setter
    def custom_monitor_names(self, value: str):
        """
        Set the custom_monitor_names to use
        :param value:
        :return:
        """
        self._custom_monitor_names = value

    @property
    def custom_monitor_thresholds(self) -> List:
        """custom_monitor_thresholds Config property"""
        return self._custom_monitor_thresholds

    @custom_monitor_thresholds.setter
    def custom_monitor_thresholds(self, value: List):
        """
        Set the custom_monitor_names to use
        :param value:
        :return:
        """
        self._custom_monitor_thresholds = value

    @property
    def custom_monitor_provider_name(self) -> List:
        """custom_monitor_provider_name Config property"""
        return self._custom_monitor_provider_name

    @custom_monitor_provider_name.setter
    def custom_monitor_provider_name(self, value: List):
        """
        Set the custom_monitor_names to use
        :param value:
        :return:
        """
        self._custom_monitor_provider_name = value

    @property
    def custom_monitor_name(self) -> List:
        """custom_monitor_name Config property"""
        return self._custom_monitor_name

    @custom_monitor_name.setter
    def custom_monitor_name(self, value: List):
        """
        Set the custom_monitor_names to use
        :param value:
        :return:
        """
        self._custom_monitor_name = value

    @property
    def custom_monitor_wml_function_provider(self) -> List:
        """custom_monitor_wml_function_provider Config property"""
        return self._custom_monitor_wml_function_provider

    @custom_monitor_wml_function_provider.setter
    def custom_monitor_wml_function_provider(self, value: List):
        """
        Set the custom_monitor_wml_function_provider to use
        :param value:
        :return:
        """
        self._custom_monitor_wml_function_provider = value

    @property
    def module(self):
        """module Config property"""
        return self._module

    @module.setter
    def module(self, value: str):
        """
        Set the custom_monitor_names to use
        :param value:
        :return:
        """
        self._module = importlib.import_module(f"mlmonitor.{value}")

    @property
    def config_file(self) -> str:
        """config_file Config property"""
        return self._config_file

    @config_file.setter
    def config_file(self, value: str):
        """
        Set the config_file to use
        :param value:
        :return:
        """
        if not value:
            value = os.path.join(
                os.path.dirname(self.module.__file__), "model_signature.json"
            )
        self._config_file = value

    @property
    def source_dir(self):
        """source_dir Config property"""
        return self._source_dir

    @source_dir.setter
    def source_dir(self, value: str):
        """
        Set the source_dir to use to instantiate Config Class
        :param value:
        :return:
        """
        self._source_dir = value

    @abstractmethod
    def save_config(self, file: str = None):
        """
        Save Model Config to JSON file to ModelConfig Object
        :param file: json file path where serialized ModelConfig Object should be saved
        :return:
        """
        pass

    @abstractmethod
    def json_object(self):
        """
        JSON for ModelConfig Object
        :param
        :return:
        """
        pass

    def _get_data_location(
        self, dataset_type: str = "train", base_data_dir: str = DATA_ROOT
    ) -> str:
        """
        returns a dataset location

        :param dataset_type:str='train': Specify whether the training or test data is to be loaded
        :param base_data_dir:str=DATA_ROOT: Specify base directory path where data is located
        :return: The location of the data
        """

        logger.debug(f"_get_data_location with dataset_type :{dataset_type}")
        assert dataset_type in {
            "train",
            "test",
            "validation",
        }, 'dataset type must be ["train","test","validation"]'
        data_path = os.path.join(base_data_dir, self.data_dir)
        if dataset_type == "train":
            fb_dataset = self.training_data
        elif dataset_type == "test":
            fb_dataset = self.test_data
        else:
            fb_dataset = self.validation_data

        # check if data_dir (last part of data_path) equals dataset_location dir
        data_location = (
            os.path.join(data_path, fb_dataset)
            if data_path.split("/")[-1] != fb_dataset
            else data_path
        )

        return data_location

    def _get_data(
        self, dataset_type: str = "train", num_samples: int = None
    ) -> Union[pd.DataFrame, Tuple]:
        """
        fetch the dataset associated with a model use case
        It takes dataset_type parameter to can be either train ,test or validation
        The function returns :
        - pandas DataFrame if structured data is fetched or (samples, labels)
        - tuple if unstructured image dataset is fetched.

        :param dataset_type:str='train': Specify whether the training or test data is to be loaded
        :param num_samples:int=None: Specify how many images should be sampled from the dataset
        :return: A pandas dataframe if the data_type is structured , tuple if unstructured image dataset
        """
        logger.debug(f"_get_data with dataset_type :{dataset_type}")
        assert dataset_type in {
            "train",
            "test",
            "validation",
        }, 'dataset type must be ["train","test","validation"]'
        assert self.data_type in [
            "structured",
            "unstructured_image",
        ], 'data_type type must be ["structured","unstructured_image"]'

        logger.debug(f"_get_data with dataset_type :{dataset_type}")

        dataset_location = self._get_data_location(dataset_type=dataset_type)

        dataset_is_csv = is_csv(dataset_location)

        if self.data_type == "structured" and dataset_is_csv:
            features = self.feature_columns
            target = self.class_label
            df = pd.read_csv(
                dataset_location, engine="python", usecols=features + [target]
            )

            if num_samples and num_samples < df.shape[0]:
                mask = random.sample(range(df.shape[0]), num_samples)
                df = df.iloc[mask, :]

            return df

        elif self.data_type == "unstructured_image":
            generate_data = getattr(
                importlib.import_module(f"mlmonitor.{self.source_dir}.utils"),
                self.fetch_data_function,
            )

            samples, labels = generate_data(
                data_path=dataset_location,
                num_samples=num_samples,
                data_type=dataset_type,
            )
            return samples, labels

    def load_config(self, file: Union[str, Dict] = None):
        """
        Load Model Config from JSON file to ModelConfig Object
        :param file: json file containing a serialized version of a ModelConfig object
        :return:
        """
        if isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
                logger.info(f"ModelConfig class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logger.info("ModelConfig class loaded from dict")
        else:
            raise ValueError("should be a file path or dictionary")

        datasets = loaded_dict.get("datasets")

        self.data_dir = datasets.get("data_dir")
        self.training_data = datasets.get("training_data")
        self.test_data = datasets.get("test_data")
        self.validation_data = datasets.get("validation_data")
        self.fetch_data_module = datasets.get("fetch_data_module")
        self.fetch_data_function = datasets.get("fetch_data_function")

        # model definition
        signature = loaded_dict.get("signature")

        self.feature_columns = signature.get("feature_columns")
        self.problem_type = signature.get("problem_type")
        self.data_type = signature.get("data_type")
        self.class_label = signature.get("class_label")
        self.prediction_field = signature.get("prediction_field")
        self.probability_fields = signature.get("probability_fields")
        self.categorical_columns = signature.get("categorical_columns")
        self.description = signature.get("description")

        self.hyperparameters = loaded_dict.get("hyperparameters")

        # Monitoring parameters
        quality_monitor = loaded_dict.get("quality_monitor")
        explain_monitor = loaded_dict.get("explain_monitor")
        fairness_monitor = loaded_dict.get("fairness_monitor")
        mrm_monitor = loaded_dict.get("mrm_monitor")
        custom_monitor = loaded_dict.get("custom_monitor")
        drift_monitor = loaded_dict.get("drift_monitor")

        # Monitoring
        self.quality_monitor_enabled = quality_monitor.get("enabled")
        self.explain_monitor_enabled = explain_monitor.get("enabled")
        self.mrm_monitor_enabled = mrm_monitor.get("enabled")
        self.custom_monitor_enabled = quality_monitor.get("enabled")
        self.fairness_monitor_enabled = fairness_monitor.get("enabled")
        self.drift_monitor_enabled = drift_monitor.get("enabled")

        self.quality_monitor_parameters = quality_monitor.get("parameters")
        self.quality_monitor_thresholds = quality_monitor.get("thresholds")

        self.fairness_monitor_parameters = fairness_monitor.get("parameters")

        self.drift_monitor_parameters = drift_monitor.get("parameters")
        self.drift_learn_constraints = drift_monitor.get("learn_constraints")

        self.custom_monitor_names = custom_monitor.get("names")
        self.custom_monitor_thresholds = custom_monitor.get("thresholds")
        self.custom_monitor_provider_name = custom_monitor.get("provider_name")
        self.custom_monitor_name = custom_monitor.get("custom_monitor_name")
        self.custom_monitor_wml_function_provider = custom_monitor.get(
            "wml_function_provider"
        )

    def valid_pyfile(self, filename: str):
        # try:
        # importlib.import_module(f"mlmonitor.{self._source_dir}.{filename[:-3]}")
        # except ModuleNotFoundError as e:
        # logger.error(f" {filename} should be a valid python module {e}")
        return bool(filename.endswith(".py"))

    def base_config_json(self) -> Dict:
        """
        returns a dictionary for Class  ModelConfig Object serialization

        :param self: Access the attributes and methods of the class in python
        :return: A dictionary use for Class Object serialization
        """

        return {
            "signature": {
                "feature_columns": self.feature_columns,
                "class_label": self.class_label,
                "prediction_field": self.prediction_field,
                "probability_fields": self.probability_fields,
                "categorical_columns": self.categorical_columns,
                "problem_type": self.problem_type,
                "data_type": self.data_type,
                "description": self.description,
            },
            "datasets": {
                "data_dir": self.data_dir,
                "training_data": self.training_data,
                "validation_data": self.validation_data,
                "test_data": self.test_data,
                "fetch_data_module": self.fetch_data_module,
                "fetch_data_function": self.fetch_data_function,
            },
            "hyperparameters": self.hyperparameters,
            "quality_monitor": {
                "enabled": self.quality_monitor_enabled,
                "parameters": self.quality_monitor_parameters,
                "thresholds": self.quality_monitor_thresholds,
            },
            "fairness_monitor": {
                "enabled": self.fairness_monitor_enabled,
                "parameters": self.fairness_monitor_parameters,
            },
            "drift_monitor": {
                "enabled": self.drift_monitor_enabled,
                "parameters": self.drift_monitor_parameters,
                "learn_constraints": self.drift_learn_constraints,
            },
            "explain_monitor": {"enabled": self.explain_monitor_enabled},
            "mrm_monitor": {"enabled": self.mrm_monitor_enabled},
            "custom_monitor": {
                "enabled": self.custom_monitor_enabled,
                "names": self.custom_monitor_names,
                "thresholds": self.custom_monitor_thresholds,
                "provider_name": self.custom_monitor_provider_name,
                "custom_monitor_name": self.custom_monitor_name,
                "wml_function_provider": self.custom_monitor_wml_function_provider,
            },
            "source_dir": self.source_dir,
        }
