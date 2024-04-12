# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Dict
import json
from typing import Union
from ibm_watson_openscale.supporting_classes.enums import DataSetTypes, TargetTypes
import pandas as pd
import os

from mlmonitor.src import logger, IAM_URL, WOS_URL, ENV
from mlmonitor.src.utils.validation import validate_uuid4
from mlmonitor.src import CATALOG_ID, MODEL_ENTRY_ID, API_KEY, DATA_ROOT

from mlmonitor.src.wml import wml_client, WML_SPACE_ID
from mlmonitor.src.wml.package import (
    create_package_extension,
    create_software_specification_extension,
    build_custmonitor_zip,
)
from mlmonitor.src.wml.deploy_custom_metrics_provider import (
    deploy_custom_metrics_provider,
)
from mlmonitor.src.wml.utils import get_deployment_uid_by_name, get_function_uid_by_name

from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.configure_wos_subscription import monitor_model
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.wos.evaluate import evaluate_monitor
from mlmonitor.src.wos.data_mart import get_datamart_ids
from mlmonitor.src.wos.configure_custom_monitor import configure_custom_monitor
from mlmonitor.src.wos.cleanup_custom_monitor import cleanup_custom_monitor
from mlmonitor.src.wos.configure_fairness_monitor import configure_fairness
from mlmonitor.src.wos.monitors import get_exising_monitors
from mlmonitor.src.wos.configure_quality_monitor import configure_quality
from mlmonitor.src.wos.configure_explain_monitor import configure_explain
from mlmonitor.src.wos.cleanup_resources import delete_deployment
from mlmonitor.src.wos.run_payload_logging import log_payload_data
from mlmonitor.src.wos.run_feedback_logging import log_feedback_data
from mlmonitor.src.demos.model_perturbator import ModelPerturbator


class ModelUseCase(ABC):
    """
    ModelUseCase Abstract Class to define a standard interface for Models Governed in CloudPak for Data.
    A ModelUseCase support several states trained,deployed,governed,monitored
    """

    def __init__(
        self,
        source_dir: str = "use_case_gcr",
        catalog_id: str = CATALOG_ID,
        model_entry_id: str = MODEL_ENTRY_ID,
        file: Union[str, Dict] = None,
    ):
        self._wos_configured = bool(wos_client)
        self._wml_configured = bool(wml_client)
        self._model_config = None

        if file:
            self.load_use_case(file=file)
        else:
            self.source_dir = source_dir
            self.catalog_id = catalog_id
            self.model_entry_id = model_entry_id
            self.model_endpoint = None
            self.is_trained = False
            self.is_deployed = False
            self.is_governed = False
            self.is_monitored = False
            self.subscription_id = None

    @property
    def serving_platform(self):
        """serving_platform Config property"""
        return self._serving_platform

    @serving_platform.setter
    def serving_platform(self, value: str):
        """
        Set the serving_platform where model should be deployed
        :param value:
        :return:
        """
        assert value in {
            "wml",
            "azure",
            "aws",
        }, f"serving_platform {value} not supported"
        self._serving_platform = value

    @property
    def model_endpoint(self) -> str:
        """ "Model Use Case model_endpoint property for Sagemaker,Azure or WML Endpoint name"""
        return self._model_endpoint

    @model_endpoint.setter
    def model_endpoint(self, value: str):
        """
        Set the model_endpoint to use
        :param value:
        :return:
        """
        self._model_endpoint = value

    @property
    def catalog_id(self) -> str:
        """Model Use Case catalog_id property for AI Factsheets Model use case"""
        return self._catalog_id

    @catalog_id.setter
    def catalog_id(self, value: str):
        """
        Set the catalog_id to use
        :param catalog_id:
        :return:
        """
        if value:
            value = value.strip('"')
            if validate_uuid4(value):
                self._catalog_id = value
            else:
                logger.warning("catalog_id should be be valid identifier")
        else:
            logger.warning("catalog_id cannot be None")

    @property
    def model_entry_id(self) -> str:
        """Model Use Case Model use case ID property for AI Factsheets Model use case"""
        return self._model_entry_id

    @model_entry_id.setter
    def model_entry_id(self, value: str):
        """
        Set the model_entry_id to use
        :param model_entry_id:
        :return:
        """
        if value:
            value = value.strip('"')
            if validate_uuid4(value):
                self._model_entry_id = value
            else:
                logger.warning("model_entry_id should be be valid identifier")
        else:
            logger.warning("model_entry_id cannot be None")

    @property
    def is_trained(self) -> bool:
        """Model Use Case Model is_trained State"""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool):
        """
        Set the Model is_trained State
        :param value: bool
        :return:
        """
        self._is_trained = value

    @property
    def is_deployed(self) -> bool:
        """Model Use Case Model is_trained State"""
        return self._is_deployed

    @is_deployed.setter
    def is_deployed(self, value: bool):
        """
        Set the Model is_deployed State
        :param value: bool
        :return:
        """
        self._is_deployed = value

    @property
    def is_governed(self) -> bool:
        """Model Use Case Model is_governed State"""
        return self._is_governed

    @is_governed.setter
    def is_governed(self, value: bool):
        """
        Set the Model is_governed State
        :param value: bool
        :return:
        """
        self._is_governed = value

    @property
    def is_monitored(self) -> bool:
        """Model Use Case Model is_monitored State"""
        return self._is_monitored

    @is_monitored.setter
    def is_monitored(self, value: bool):
        """
        Set the Model is_monitored State
        :param value: bool
        :return:
        """
        self._is_monitored = value

    @property
    def subscription_id(self) -> str:
        """ ""Model Use Case subscription_id for Watson OpenScale subscription created as a result of monitor method"""
        return self._subscription_id

    @subscription_id.setter
    def subscription_id(self, value: str):
        """
        Set the subscription_id to use
        :param subscription_id:
        :return:
        """
        if value:
            value = value.strip('"')
            if validate_uuid4(value):
                self._subscription_id = value
            else:
                logger.warning("subscription_id should be be valid identifier")
        self._subscription_id = value

    @abstractmethod
    def save_use_case(self, file: str = None):
        """
        Save Model use Case to JSON file to ModelUseCase Object
        :param file: json file path where serialized ModelUseCase Object should be saved
        :return:
        """
        pass

    def display_states(self):
        print(
            f"\n\
              - trained:   [{self.is_trained}]\n\
              - deployed:  [{self.is_deployed}]\n\
              - governed:  [{self.is_governed}]\n\
              - monitored: [{self.is_monitored}]\n"
        )

    def load_use_case(self, file: Union[str, Dict]):
        """
        Instantiate ModelUseCase Object from a file or a python dictionary

        :param self: Reference the class instance
        :param file:Union[str,Dict] Specify the file or Dictionary to load
        :return:
        """

        if isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
                logger.info(f"ModelUseCase class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logger.info("ModelUseCase class loaded from dict")
        else:
            raise ValueError("should be a file path or dictionary")

        self.model_state = loaded_dict.get("model_state")
        self.model_endpoint = loaded_dict.get("model_endpoint")
        self.subscription_id = loaded_dict.get("subscription_id")
        self.source_dir = loaded_dict.get("source_dir")
        self.catalog_id = loaded_dict.get("catalog_id")
        self.model_entry_id = loaded_dict.get("model_entry_id")
        self.is_trained = loaded_dict.get("is_trained")
        self.is_deployed = loaded_dict.get("is_deployed")
        self.is_governed = loaded_dict.get("is_governed")
        self.is_monitored = loaded_dict.get("is_monitored")
        self.serving_platform = loaded_dict.get("serving_platform")

    @abstractmethod
    def json_object(self):
        """
        JSON for ModelUseCase Object
        :param
        :return:
        """
        pass

    @abstractmethod
    def derive_model_states(self, endpoint_name: str):
        """
        Derive model state
        :param: endpoint_name from which Object should be populated
        :return: Dictionary of model resource created
        """
        pass

    @abstractmethod
    def train(self):
        """
        Perform model training in target model serving platform
        :param
        :return: Dictionary of model resource created
        """
        pass

    @abstractmethod
    def _train_local(self):
        """
        Perform model training on local environment
        :param
        :return: Dictionary of model resource created
        """
        pass

    @abstractmethod
    def deploy(self):
        """
        Perform model deployment in target model serving platform
        :param
        :return: Dictionary of model resource created
        """
        pass

    @abstractmethod
    def govern(self):
        """
        Perform model deployment metadata capture in AI FactSheets
        :param
        :return: Dictionary of model resource created
        """
        pass

    def monitor(self):
        """
        It takes an instance of a ModelConfig class for a deployed model (must be trained AND deployed in serving platform)
         and uses it to create a Watson OpenScale subscription with monitors

        :param self: Access fields or methods of the class in python
        :return:
        """
        if not self.model_endpoint:
            raise ValueError("No Model Endpoint assigned to this Object")

        if not self._is_endpoint_deployed():
            raise ValueError(
                f"{self.serving_platform.upper()} model endpoint {self.model_endpoint} not deployed cannot configure monitoring"
            )

        subscription_ids = get_subscription_id_by_deployment(
            wos_client=wos_client, deployment_name=self.model_endpoint
        )

        if len(subscription_ids) > 0:
            logger.warning(
                f"{self.model_endpoint} has already at least one WOS subscription ID => {len(subscription_ids)} found {subscription_ids}"
            )
            subscription_id = subscription_ids[0]
        else:
            # ! TODO support non production
            provider_type = "production"
            subscription_id = monitor_model(
                model_config=self._model_config,
                deployment_name=self.model_endpoint,
                deployment_target=self.serving_platform,
                wos_provider_type=provider_type,
                wos_provider_name=f"{self.serving_platform}_{provider_type}".upper(),
                wos_provider_description=f"{self.serving_platform.upper()} ML Model monitoring",
                wos_provider_keep=True,
                data_path=os.path.join(DATA_ROOT, self._model_config.data_dir),
            )

        self.is_monitored = True
        self.subscription_id = subscription_id

    def configure_fairness_monitor(self):
        """
        Configure a Fairness Monitor for the Watson OpenScale subscription associated with this model use case
        :param
        :return: Dictionary of model resource created
        """
        if not self.subscription_id:
            logger.warning(f"No Subscription ID for {self.model_endpoint}")
            return {}

        wos_config_params = {
            "model_config": self._model_config,
            "deployment_name": self.model_endpoint,
            "keep_wos_monitor": True,
        }
        fairness_monitor_id = configure_fairness(**wos_config_params)
        logger.debug(f"Fairness Monitor Ready {fairness_monitor_id}")
        existing_monitors = get_exising_monitors(
            wos_client=wos_client, subscription_id=self.subscription_id
        )

        return existing_monitors

    def configure_drift_monitor(self):
        """
        Configure a Drift Monitor for the Watson OpenScale subscription associated with this model use case
        :param
        :return: Dictionary of model resource created
        """
        try:
            import ibm_wos_utils

            logger.info(f"ibm_wos_utils installed at {ibm_wos_utils.__path__}")
            drift_trainer = True
        except ModuleNotFoundError:
            logger.warning("run pip install mlmonitor[drift] to configure drift")
            drift_trainer = False

        if not self.subscription_id:
            logger.warning(f"No Subscription ID for {self.model_endpoint}")
            return {}

        os.environ["model_endpoint"] = self.model_endpoint

        wos_config_params = {
            "model_config": self._model_config,
            "deployment_name": self.model_endpoint,
            "keep_wos_monitor": True,
            "deployment_target": self.serving_platform,
            "data_path": os.path.join(DATA_ROOT, self._model_config.data_dir),
        }

        if self._model_config.drift_monitor_enabled and drift_trainer:
            from mlmonitor.src.wos.configure_drift_monitor import configure_drift

            drift_monitor_id = configure_drift(**wos_config_params)
            logger.debug(f"Drift Monitor Ready {drift_monitor_id}")
        else:
            logger.warning(
                "run pip install mlmonitor[drift] to be able to configure drift monitor"
            )

        existing_monitors = get_exising_monitors(
            wos_client=wos_client, subscription_id=self.subscription_id
        )

        return existing_monitors

    def configure_custom_monitor(self, version: str = "0.1") -> dict:
        """
        Configure a Custom Monitor for the Watson OpenScale subscription associated with this model use case
        :param
        :return: Dictionary of model resource created
        """
        # Build package with custom monitor helpers and get_metrics code
        pkg_name, pkg_path = build_custmonitor_zip(version=version)
        # Create WML runtime with these custom monitor helpers installed
        pkg_extn_uid = self._create_custom_monitor_package_extension(
            pkg_extn_name=pkg_name, pkg_extn_path=pkg_path
        )
        # package_extension and sw_specification are assigned the same name
        self._create_custom_monitor_sw_specification(
            pkg_extn_uid=pkg_extn_uid, sw_spec_name=pkg_name
        )
        # Deploy WML function (custom metrics provider) able to handle Custom monitor evaluation requests (using sw_specification runtime)
        self._deploy_custom_metrics_provider(runtime=pkg_name)

        # Create Watson OpenScale Custom Monitor , integrated with WML custom metrics provider
        custom_monitor_config = {
            "monitored_model": self.model_endpoint.strip(),
            "wos_client": wos_client,
            "wml_client": wml_client,
            "deployment_name": f"{self._model_config.custom_monitor_wml_function_provider}-deploy",
            # WML function  name deployed for this custom monitor
            "provider_name": self._model_config.custom_monitor_provider_name,
            "custom_monitor_name": self._model_config.custom_monitor_name,  # Name Displayed in WOS UI
            "custom_metrics_names": tuple(self._model_config.custom_monitor_names),
            "custom_metrics_thresholds": tuple(
                self._model_config.custom_monitor_thresholds
            ),
            "wml_space_id": WML_SPACE_ID,
            "apikey": API_KEY,
            "auth_url": IAM_URL,
        }

        custom_monitor_instance_details = configure_custom_monitor(
            **custom_monitor_config
        )

        logger.info(
            f"custom monitor created  : {json.dumps(custom_monitor_instance_details.to_dict(), indent=4)}"
        )

        evaluate_monitor(
            deployment_name=self.model_endpoint.strip(),
            monitor_types=(self._model_config.custom_monitor_name.strip().lower(),),
        )

        return custom_monitor_instance_details

    def delete_custom_monitor(self):
        """
        deletes a custom monitor.
        does the following:
            1) Deletes WML deployment function that was created for this custom monitor (custom metrics provider).
            2) Deletes the WML function that was created for this custom monitor.
            3) Deletes monitor instance in WOS

        :return: None
        """

        data_marts = get_datamart_ids(wos_client=wos_client)

        if len(data_marts) != 1:
            raise ValueError(f"Please Specify datamart to use among {data_marts}")

        data_mart_id = data_marts[0]

        # CUSTOM MONITOR SPECIFIC NAMES
        provider_name = self._model_config.custom_monitor_name
        # Name Displayed in WOS UI
        custom_monitor_name = self._model_config.custom_monitor_provider_name
        wml_function_provider = self._model_config.custom_monitor_wml_function_provider

        # Deployment name corresponds to the WML function deployed for this custom monitor
        deployment_name = f"{wml_function_provider}-deploy"
        py_fname = f"{wml_function_provider}-function"

        cleanup_custom_monitor(
            wos_client=wos_client,
            provider_name=provider_name,
            custom_monitor_name=custom_monitor_name,
            subscription_id=self.subscription_id,
            data_mart_id=data_mart_id,
        )

        if deployment_uid := get_deployment_uid_by_name(
            wml_client=wml_client, deployment_name=deployment_name
        ):
            wml_client.deployments.delete(deployment_uid=deployment_uid)

        if function_uid := get_function_uid_by_name(
            wml_client=wml_client, function_name=py_fname
        ):
            wml_client.repository.delete(artifact_uid=function_uid)

    def configure_quality_monitor(self):
        """
        Configure a Quality Monitor for the Watson OpenScale subscription associated with this model use case
        :param
        :return: Dictionary of model resource created
        """
        if not self.subscription_id:
            logger.warning(f"No Subscription ID for {self.model_endpoint}")
            return {}

        wos_config_params = {
            "model_config": self._model_config,
            "deployment_name": self.model_endpoint,
            "keep_wos_monitor": True,
            "data_mart_id": None,
        }

        quality_monitor_id = configure_quality(**wos_config_params)
        logger.debug(f"Quality Monitor Ready {quality_monitor_id}")
        existing_monitors = get_exising_monitors(
            wos_client=wos_client, subscription_id=self.subscription_id
        )

        return existing_monitors

    def configure_explainability_monitor(self):
        """
        Configure Explainability Monitor for the Watson OpenScale subscription associated with this model use case
        :param:
        :return: Dictionary of model resource created
        """
        if not self.subscription_id:
            logger.warning(f"No Subscription ID for {self.model_endpoint}")
            return {}

        wos_config_params = {
            "model_config": self._model_config,
            "deployment_name": self.model_endpoint,
            "keep_wos_monitor": True,
        }

        explain_monitor_id = configure_explain(**wos_config_params)
        logger.debug(f"Explainability Monitor Ready {explain_monitor_id}")
        existing_monitors = get_exising_monitors(
            wos_client=wos_client, subscription_id=self.subscription_id
        )

        return existing_monitors

    def log_payload(self, num_samples: int = 100, dataset_type: str = "validation"):
        """
        logs a payload to Watson OpenScale subscription for a given endpoint associated to a model use case
        :param num_samples:int=100: Define the number of samples to be used for payload logging
        :param dataset_type:str=validation: Define the type of dataset to read from for payload logging
        :return: None
        """

        payload_data_set_id = (
            wos_client.data_sets.list(
                type=DataSetTypes.PAYLOAD_LOGGING,
                target_target_id=self.subscription_id,
                target_target_type=TargetTypes.SUBSCRIPTION,
            )
            .result.data_sets[0]
            .metadata.id
        )

        if payload_data_set_id is None:
            raise RuntimeError(
                "Payload dataset not found. Please check subscription status."
            )
        else:
            logger.info(f"Payload data set id: {payload_data_set_id}")

        inference_samples = (
            min(num_samples, 10)
            if self.source_dir == "use_case_mnist_tf"
            else num_samples
        )
        log_payload_data(
            model_config=self._model_config,
            deployment_name=self.model_endpoint,
            deployment_target=self.serving_platform,
            inference_samples=inference_samples,
            dataset_type=dataset_type,
        )

    def log_feedback(self, num_samples: int = 100, dataset_type: str = "validation"):
        """
        Perform Feedback logging to Watson OpenScale for this Model Use Case
        :param num_samples:int=100: Define the number of samples to be used for feedback logging
        :param dataset_type:str=validation: Define the type of dataset to read from for feedback logging
        :return: Dictionary of model resource created
        """
        assert dataset_type in {"train", "test", "validation"}
        assert (
            self.subscription_id
        ), "Model use case must be monitored to perform feedback logging"
        feedback_data_set_id = wos_client.data_sets.list(
            type=DataSetTypes.FEEDBACK,
            target_target_id=self.subscription_id,
            target_target_type=TargetTypes.SUBSCRIPTION,
        ).result.data_sets

        if len(feedback_data_set_id) > 0:
            logger.info(f"Feedback data set id:{feedback_data_set_id}")

            include_predictions = (
                self.source_dir == "use_case_mnist_tf"
                and self.serving_platform == "aws"
            )
            inference_samples = (
                min(num_samples, 10)
                if self.source_dir == "use_case_mnist_tf"
                else num_samples
            )

            log_feedback_data(
                model_config=self._model_config,
                deployment_name=self.model_endpoint,
                deployment_target=self.serving_platform,
                inference_samples=inference_samples,
                include_predictions=include_predictions,
                dataset_type=dataset_type,
            )
        else:
            logger.warning(
                "log_feedback can only be invoked when Openscale subscription has a Feedback data set => "
                "run configure_quality_monitor()"
            )

    @abstractmethod
    def score_model(self):
        """
        Perform scoring request to deployed model
        :param
        :return: Dictionary of model resource created
        """
        pass

    def evaluate_model(self):
        """
        Perform a Model Evaluation for all existing monitors for Watson OpenScale
        subscription associated to this model use case
        :param
        :return: Dictionary of model resource created
        """
        existing_monitors = get_exising_monitors(
            wos_client=wos_client, subscription_id=self.subscription_id
        )
        evaluate_monitor(
            deployment_name=self.model_endpoint,
            monitor_types=tuple(existing_monitors.keys()),
        )

    def cleanup(self) -> Dict:
        """
        delete all resources associated to a model use case (deployment, AI Fatcsheets assets, Watson OpenScale subscription)

        :param self: Access fields or methods of the class in python
        :return: Dictionary of status for each deleted resource : Model serving (Sagemaker,WML,Azure...)  , WOS , Factsheets model asset and count of Number of resources deleted
        """
        deleted_items = delete_deployment(
            deployment_name=self.model_endpoint,
            model_entry_id=self.model_entry_id,
            catalog_id=self.catalog_id,
            apikey=API_KEY,
            deployment_target=self.serving_platform,
        )
        logger.info(
            f"deleted items for model use case {self.model_endpoint}: {deleted_items}"
        )
        self.is_deployed = False
        self.is_governed = False
        self.is_monitored = False
        self.subscription_id = None
        self.model_endpoint = None
        self._reset_states()

        return deleted_items

    @abstractmethod
    def perturb_scoring(self, dataset_type="train", **kwargs) -> pd.DataFrame:
        """
        perturb single column of dataset of type dataset_type
        :param: dataset_type:str dataset to use before adding perturbation
        :return: pandas dataframe with perturbed column
        """
        pass

    def data_drift_scenario(
        self,
        model_perturbation: ModelPerturbator = None,
        dataset_type: str = "train",
    ):
        """
        Perform Payload logging with perturbed data to demo data drift scenario on Watson OpenScale
        :param: col:str column to be perturbed
        :param: dataset_type:str dataset to use for data drift scenario
        :return: Dictionary of model resource created
        """

        # Verifications
        assert dataset_type in {"test", "train", "validation"}
        assert self.source_dir in [
            "use_case_churn",
            "use_case_gcr",
        ], "data_drift_scenario supported on for churn prediction and credit risk use cases"
        assert (
            self.model_endpoint
        ), "data_drift_scenario No Model Endpoint assigned to this Object"
        assert (
            self._is_endpoint_deployed()
        ), f"data_drift_scenario No Model Endpoint deployed with name : {self.model_endpoint}"
        assert self.subscription_id, f"No Subscription ID for {self.model_endpoint}"
        assert (
            self._model_config.data_type == "structured"
        ), f"Only structured data types supported for data_drift_scenario : [{self._model_config.data_type}]"

        # Predefined drift scenario
        if not model_perturbation:
            monitor_type = "drift"
            scenario_id = "double_column_1"

            model_perturbation = ModelPerturbator(
                source_dir=self.source_dir,
                monitor_type=monitor_type,
                scenario_id=scenario_id,
            )

        perturbation_args = model_perturbation.model_perturbation_json()
        self.perturb_scoring(dataset_type, **perturbation_args)

    @abstractmethod
    def _is_endpoint_deployed(self) -> bool:
        """
        assert if model endpoint is deployed
        :param
        :return: Dictionary of model resource deleted
        """
        pass

    @abstractmethod
    def _reset_states(self):
        """
        reset model use case states
        :param
        :return: None
        """
        pass

    def _deploy_custom_metrics_provider(self, runtime: str) -> str:
        """
        deploys a custom metrics provider function to WML using `publish` function of custom monitor helpers to abstract custom monitor workflow

        :param runtime:str: custom monitor software extension runtime
        :return: deployment_uid of the custom metrics provider
        """

        data_marts = get_datamart_ids(wos_client=wos_client)

        if len(data_marts) != 1:
            raise ValueError(f"Please Specify datamart to use among {data_marts}")

        data_mart_id = data_marts[0]

        wml_function_provider = self._model_config.custom_monitor_wml_function_provider
        # Deployment name corresponds to the WML function deployed for this custom monitor
        deployment_name = f"{wml_function_provider}-deploy"
        py_fname = f"{wml_function_provider}-function"

        logger.info("Deploy WML function for custom monitor")

        username = "default_user"
        if ENV == "prem":
            from mlmonitor.src import USERNAME

            username = USERNAME

        def custom_metrics_provider(
            url=WOS_URL,
            apikey=API_KEY,
            use_case=self.source_dir,
            env=ENV,
            username=username,
        ):
            """
            wrapper function to create a WML function:
                url (str): URL of Watson OpenScale service.
                apikey (str): WOS API Key.
                use_case (str): valid model use case name to determine which get_metrics function should be invoked

            :param url: Watson OpenScale URL
            :param apikey: API Key to authenticate with Watson OpenScale service
            :param use_case: valid model use case
            :param env:str = type of environment where custom metrics provider is deployed 'saas' or 'prem'
            :param username:str = username of CP4D user only used if 'prem' environment is selected
            :return: A function that can be used to publish metrics to the custom monitor and executed as WML deployment
            """
            import importlib
            from custmonitor.metricsprovider.helpers import publish

            get_metrics = getattr(
                importlib.import_module(f"custmonitor.metrics.{use_case}"),
                "get_metrics",
            )

            def publish_to_monitor(input_data):
                response_payload = publish(
                    input_data=input_data,
                    url=url,
                    apikey=apikey,
                    get_metrics_fn=get_metrics,
                    env=env,
                    username=username,
                )
                return response_payload

            return publish_to_monitor

        deploy_custom_metrics_provider(
            deployment_name=deployment_name,
            function_code=custom_metrics_provider,
            wml_space_id=WML_SPACE_ID,
            python_function_name=py_fname,
            runtime=runtime,
        )

        input_data = {
            "input_data": [
                {
                    "values": {
                        "data_mart_id": data_mart_id,
                        "subscription_id": self.subscription_id,
                        "test": "test",
                        "custom_monitor_run_id": "123",
                        "custom_monitor_id": "not needed",
                        "custom_monitor_instance_id": "not needed",
                        "custom_monitor_instance_params": {
                            "custom_metrics_provider_id": "not needed",
                            "custom_metrics_wait_time": 300,
                        },
                    }
                }
            ]
        }

        deployment_uid = get_deployment_uid_by_name(
            wml_client=wml_client, deployment_name=deployment_name
        )
        res = wml_client.deployments.score(deployment_uid, input_data)
        logger.info(
            f"scoring custom metrics provider {deployment_name}:\n{json.dumps(res, indent=4)}"
        )
        return deployment_uid

    @staticmethod
    def _create_custom_monitor_package_extension(
        pkg_extn_path: str, pkg_extn_name: str
    ) -> str:
        """
        creates a custom package extension for the Custom Monitors in WML
        returns the uid of the package extension, If a package extension exists with this version, it will be deleted and
        recreated.

        :param version:str=0.5: Specify the version of the package extension
        :return: The uid of the package extension
        """
        pkg_extn_uid = wml_client.package_extensions.get_uid_by_name(pkg_extn_name)

        if pkg_extn_uid != "Not Found":
            wml_client.package_extensions.delete(pkg_extn_uid)

        if os.path.exists(pkg_extn_path):

            pkg_extn_description = "Pkg extension for Custom Monitor helpers"
            pkg_extn_type = "pip_zip"
            pkg_extn_uid, pkg_extn_url, details = create_package_extension(
                wml_client,
                pkg_extn_name,
                pkg_extn_description,
                pkg_extn_path,
                pkg_extn_type,
            )

            logger.info(
                f"pkg_extn_uid : {pkg_extn_uid}, "
                f"pkg_extn_url : {pkg_extn_url}, "
                f"pkg_extn_details:\n{json.dumps(details, indent=4)}"
            )
        else:
            details = wml_client.package_extensions.get_details(pkg_extn_uid)
            raise ValueError(f"{pkg_extn_path} not found with details:\n{details}")

        return pkg_extn_uid

    @staticmethod
    def _create_custom_monitor_sw_specification(
        pkg_extn_uid: str,
        sw_spec_name: str,
        base_sw_spec: str = "runtime-23.1-py3.10",
    ) -> str:
        """
        creates a software specification for the custom monitor that will be used to deploy custom metrics provider in WML.

        :param pkg_extn_uid:str: package extension that is used to create the custom metrics provider python function and deployment
        :param sw_spec_name:str: unique name for the software specification
        :param base_sw_spec:str=runtime-23.1-py3.10: Specify the base software specification to use as a starting point
        :return: software specification uid after creation
        """

        sw_sepc_decr = f"Software specification with {sw_spec_name}"

        sw_spec_uid = wml_client.software_specifications.get_uid_by_name(sw_spec_name)

        if sw_spec_uid != "Not Found":
            wml_client.software_specifications.delete(sw_spec_uid=sw_spec_uid)

        sw_spec_uid = create_software_specification_extension(
            wml_client, pkg_extn_uid, sw_spec_name, sw_sepc_decr, base_sw_spec
        )
        logger.info(f"SW spec {sw_spec_name} created with ID {sw_spec_uid}\n")
        return sw_spec_uid

    def base_use_case_json(self) -> Dict:
        return {
            "serving_platform": self.serving_platform,
            "source_dir": self.source_dir,
            "model_endpoint": self.model_endpoint,
            "catalog_id": self.catalog_id,
            "model_entry_id": self.model_entry_id,
            "subscription_id": self.subscription_id,
            "is_trained": self.is_trained,
            "is_deployed": self.is_deployed,
            "is_governed": self.is_governed,
            "is_monitored": self.is_monitored,
        }
