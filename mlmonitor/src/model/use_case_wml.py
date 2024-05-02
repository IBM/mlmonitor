# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Union
import json
import os
import importlib
from random import randint
import time

from mlmonitor.src import (
    API_KEY,
    ENV,
    CATALOG_ID,
    MODEL_ENTRY_ID,
    logger,
    PROJECT_ROOT,
    DATA_ROOT,
    MODEL_ROOT,
)

from mlmonitor.src.model.config_wml import WMLModelconfig
from mlmonitor.src.model.use_case import ModelUseCase
from mlmonitor.src.wml import wml_client, WML_SPACE_ID, WML_URL
from mlmonitor.src.wml.utils import get_deployment_uid_by_name, get_model_uid_by_name
from mlmonitor.src.wml.deploy_wml_endpoint import (
    deploy_wml_endpoint,
    create_model_asset,
)
from mlmonitor.src.wml.scoring import get_wos_response, _score_unstructured
from mlmonitor.src.wos.run_payload_logging import log_as_wos_payload
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.demos.scenario_helpers import perturb_column, perturb_double_column

from mlmonitor.use_case_churn.utils import (
    git_branch,
)  # TODO put this outside use_case_churn
from mlmonitor.src.wos.evaluate import evaluate_monitor


class WMLModelUseCaseEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, WMLModelUseCase):
            return obj.json_object()
        return super().default(obj)


class WMLModelUseCase(ModelUseCase):
    """
    ModelUseCase trained and deployed in AWS WML
    governed in CloudPak for Data AI FactSheets
    monitored in Watson OpenScale.

    _model_config class attribute from is a WMLModelConfig Object that contain all the relevant parameters
    to bring a WMLModelUseCase instance to production (monitor state)

    supported states are :

        - "trained"
        - "deployed"
        - "governed"
        - "monitored"

    """

    def __init__(
        self,
        source_dir: str,
        catalog_id: str = CATALOG_ID,
        model_entry_id: str = MODEL_ENTRY_ID,
        file: Union[str, Dict] = None,
    ):

        super().__init__(
            source_dir=source_dir,
            catalog_id=catalog_id,
            model_entry_id=model_entry_id,
            file=file,
        )

        self.load_wml_use_case(file=file)

    @property
    def model_data(self) -> str:
        """Model Use Case property for model binary location in S3 bucket"""
        return self._model_data

    @model_data.setter
    def model_data(self, value: str):
        """
        Set the model_data to use
        :param model_data:
        :return:
        """
        self._model_data = value

    @property
    def model_uid(self) -> str:
        """Model ID once a WML model is published as an Asset in a Deployment Space"""
        return self._model_uid

    @model_uid.setter
    def model_uid(self, value: str):
        """
        Set the model_uid to use
        :param model_uid:
        :return:
        """
        self._model_uid = value

    def save_use_case(self, file: str = None):
        """
        Save WMLModelUseCase to JSON file to WMLModelUseCase Object
        :param file: json file path where serialized WMLModelUseCase Object should be saved
        :return:
        """
        if not file:
            file = os.path.join("./", "WMLModelUseCase.json")
        logger.info(f"WMLModelUseCase class saved to {file}")
        with open(file, "w") as f:
            json.dump(obj=self, fp=f, cls=WMLModelUseCaseEncoder, indent=4)

    def load_wml_use_case(self, file: Union[str, Dict] = None):
        """
        Instantiate WMLModelUseCase Object from a file or a python dictionary

        :param self: Reference the class instance
        :param file:Union[str,Dict] Specify the file or Dictionary to load
        :return:
        """

        if file is None:
            loaded_dict = {}
            logger.info("WMLModelUseCase class initialized from scratch")
        elif isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
                logger.info(f"WMLModelUseCase class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logger.info("WMLModelConfig class loaded from dict")
        else:
            raise ValueError("should be a file path or dictionary")

        self.serving_platform = "wml"
        self._df = None
        self.model_data = loaded_dict.get("model_data")
        self.model_uid = loaded_dict.get("model_uid")
        self._model_config = WMLModelconfig(
            source_dir=self.source_dir,
            file=loaded_dict.get("model_config", None),  # = model_signature.json
        )
        # assign a model endpoint name
        self.model_endpoint = loaded_dict.get(
            "model_endpoint", self._assign_model_endpoint()
        )

    def json_object(self):
        """
        returns a json object of WMLModelUseCase class that can be used for serialization and deserialization of the Object.
        :param self: Access the class attributes
        :return: A dictionary is used by WMLModelUseCaseEncoder. This dictionary is compatible to load a WMLModelUseCase Object
        """
        BaseUseCaseObject = self.base_use_case_json()
        BaseUseCaseObject["__class__"] = "WMLModelUseCase"
        # update model config related to WML
        WMLModelUseCaseObject = self.wml_use_case_json()

        WMLModelConfigObject = self._model_config.json_object()

        return {
            **BaseUseCaseObject,
            **WMLModelUseCaseObject,
            **{"model_config": WMLModelConfigObject},
        }

    def derive_model_states(self, endpoint_name: str = None):
        """
        infer model states from a given model endpoint
        :param: endpoint_name from which model use case states should be inferred
        :return:
        """

        # reset all states
        self.model_endpoint = endpoint_name
        self.subscription_id = None

        self.is_trained = False
        self.is_deployed = False
        self.is_governed = False
        self.is_monitored = False
        # CHECK WML Deployments
        wml_client.set.default_space(WML_SPACE_ID)
        if self._is_endpoint_deployed():
            self._assign_params_from_Endpoint(ep_name=endpoint_name)
        else:
            self.model_uid = get_model_uid_by_name(
                wml_client=wml_client, model_name=endpoint_name
            )
        self._assign_params_from_model()
        # CHECK FOR Subscription(s) in Watson OpenScale
        if wos_client:  # wos must be instantiated
            if (
                len(
                    subscription_ids := get_subscription_id_by_deployment(
                        wos_client=wos_client, deployment_name=endpoint_name
                    )
                )
                > 1
            ):
                raise ValueError(
                    f"{self.model_endpoint} has already more than one WOS subscription ID => {len(subscription_ids)} found {subscription_ids}"
                )
            elif len(subscription_ids) == 0:
                logger.info(
                    f"No Model Monitoring found for [{endpoint_name}] in OpenScale"
                )
            else:
                logger.info(
                    f"1 Model Monitoring Subscription found for [{endpoint_name}] in OpenScale"
                )
                self._assign_params_from_WOS(subscription_id=subscription_ids[0])

    def init_internal_fs_client(self, logger, **params) -> tuple:
        """
        initialize the factsheet client for internal model

        :param logger: instance of a logging object

        :returns:
            Tuple of factsheet client, model properties, experiment name, experiment ID, and factsheet tags
        """
        from ibm_aigov_facts_client import (
            AIGovFactsClient,
            ModelEntryProps,
            CloudPakforDataConfig,
        )

        experiment_name = self.model_endpoint

        aigov_params = {
            "experiment_name": experiment_name,
            "set_as_current_experiment": True,
            "enable_autolog": True,
            "external_model": False,
            "container_type": "space",
            "container_id": WML_SPACE_ID,
        }

        if ENV == "saas":
            aigov_params["api_key"] = API_KEY
        elif ENV == "prem":
            from mlmonitor.src import USERNAME

            aigov_params["cloud_pak_for_data_configs"] = CloudPakforDataConfig(
                service_url=WML_URL, username=USERNAME, api_key=API_KEY
            )
        else:
            raise ValueError(
                f"ENV set to '{ENV}'.Value should be set to 'saas' (IBM Cloud) or 'prem' (On premise cluster)"
            )

        props = ModelEntryProps(
            model_entry_catalog_id=params.get("catalog_id"),
            model_entry_id=params.get("model_entry_id"),
        )

        facts_client = AIGovFactsClient(**aigov_params)

        experiment_id = facts_client.experiments.get_current_experiment_id()
        logger.info(f"Current Experiment {experiment_name} ID {experiment_id}")

        # Capturing tags
        fs_tags = {"git_branch": git_branch(logger=logger, path=PROJECT_ROOT)}

        return facts_client, props, experiment_name, experiment_id, fs_tags

    def train(self):
        """
        train WML model based on model use case definition for inference (runtime,script...) in wml_runtime section of model signature
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, job name and state of training
        """
        self._train_local()

    def _train_local(self):
        """
        train WML model in local environment based on model use case definition
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, job name and state of training
        """
        train = getattr(
            importlib.import_module(
                f"mlmonitor.{self._model_config.source_dir}.{self._model_config.train_module}"
            ),
            self._model_config.train_method,
        )

        parameters = self._model_config.hyperparameters

        # Specifying endpoints if not already assigned
        if not self.model_endpoint:
            self.model_endpoint = self._assign_model_endpoint()

        (
            facts_client,
            model_props,
            _,
            experiment_id,
            fs_tags,
        ) = self.init_internal_fs_client(
            logger=logger,
            catalog_id=self.catalog_id,
            model_entry_id=self.model_entry_id,
        )

        # Dictionary of metrics, dict of params, dict of tags
        model_data = train(
            model_dir=MODEL_ROOT,
            data_path=os.path.join(DATA_ROOT, self._model_config.data_dir),
            train_dataset=self._model_config.training_data,
            val_dataset=self._model_config.validation_data,
            test_dataset=self._model_config.test_data,
            logger=logger,
            **parameters,
        )

        self.is_trained = True
        self.model_data = model_data

        # Publish model
        published_model_details = create_model_asset(
            fs_client=facts_client,
            model_data=self.model_data,
            deployment_name=self.model_endpoint,
            model_config=self._model_config,
        )

        self.model_uid = published_model_details

        run_id = facts_client.runs.get_current_run_id()

        if len(fs_tags) > 0:
            facts_client.runs.set_tags(run_id, fs_tags)
            logger.debug(
                f"save_fs_model Factsheets tags :\n{json.dumps(fs_tags, indent=4)}"
            )

        # Exporting facts
        logger.info(f"Current Experiment ID {experiment_id}")
        runs = facts_client.runs.list_runs_by_experiment(experiment_id)
        logger.info(f"Runs :\n{runs}")
        # facts_client.export_facts.export_payload(run_id)
        facts_client.export_facts.export_payload(run_id)

        # Get WML model and track it
        wml_model = facts_client.assets.get_model(
            model_id=published_model_details
        )  # wml_stored_model_details=?

        muc_utilities = facts_client.assets.get_ai_usecase(
            ai_usecase_id=self.model_entry_id,
            catalog_id=self.catalog_id,
        )

        wml_model.track(
            usecase=muc_utilities,
            approach=muc_utilities.get_approaches()[0],
            version_number="minor",  # "0.1.0"
        )

    def deploy(self):
        """
        deploy WML model based on model use case definition for inference (runtime,script...).
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, deployment details
        """

        if self.model_data:

            if not self._is_endpoint_deployed():
                deploy_wml_endpoint(
                    deployment_name=self.model_endpoint,
                    model_config=self._model_config,
                    model_uid=self.model_uid,
                )
            else:
                logger.warning(
                    f"WML model endpoint {self.model_endpoint} already deployed"
                )
            self.is_deployed = True
        else:
            logger.error(
                "Model Endpoint cannot be deployed without valid model data location `model_data` attribute"
            )

    def govern(self):
        """
        not needed for WML models
        """
        if not self.model_data:
            raise ValueError("Model data location should be set")

        if not self.model_endpoint:
            raise ValueError("Model endpoint name should be set")

        self.is_governed = True

    def score_model(self):
        """
        Perform scoring request to deployed WML model (structured or unstructured datasets)
        :param
        :return: model predictions
        """
        if not self._is_endpoint_deployed():
            logger.warning(f"WML model endpoint {self.model_endpoint} not deployed")
            return {"fields": {}, "values": {}}

        inference_samples = 2
        test_data = self._model_config._get_data(
            dataset_type="test", num_samples=inference_samples
        )

        if self._model_config.data_type == "structured":

            df = test_data.loc[:, self._model_config.feature_columns]

            response_wos = get_wos_response(
                df=df, endpoint_name=self.model_endpoint, wml_client=wml_client
            )

        elif self._model_config.data_type == "unstructured_image":

            samples, labels = test_data
            print("samples shape sent for inference", samples.shape)

            result = _score_unstructured(
                payload=samples, endpoint_name=self.model_endpoint, client=wml_client
            )

            predictions = result.get("predictions")[0].get("values")
            fields = result.get("predictions")[0].get("fields")
            # scoring_input_target = [ [x, y] for x, y in zip(samples.tolist(), labels.tolist()) ]
            print("predictions:", [x[0] for x in predictions])
            print("target     :", labels.tolist())
            response_wos = {"fields": fields, "values": predictions}

        else:
            raise ValueError(
                "supported data_type are structured or unstructured_image (must be passed in model signature)"
            )

        logger.info(
            f"WML Endpoint {self.model_endpoint} scoring response:\n{response_wos}"
        )

        return response_wos

    @log_as_wos_payload("wml")
    def send_perturbation(self, dataset_type="train", **kwargs):

        if "source_column" in kwargs and "source_cond" in kwargs:
            return perturb_double_column(df=self._df, **kwargs)

        else:
            return perturb_column(df=self._df, **kwargs)

    def perturb_scoring(self, dataset_type="train", **kwargs):
        ratios_list = kwargs.pop("ratios")

        for ratio in ratios_list:
            kwargs["ratio"] = ratio
            self.send_perturbation(dataset_type, **kwargs)

            evaluate_monitor(
                deployment_name=self.model_endpoint,
                monitor_types=("drift",),  # TODO why tuple?
            )

    def wml_use_case_json(self) -> Dict:
        """
        returns a dictionary containing the job_name, catalog_id, and model_endpoint of a SageMaker Model Use case.

        :param self: Access the class attributes
        :return: dictionary with job_name, catalog_id and model_endpoint as keys
        """
        return {
            "catalog_id": self.catalog_id,
            "model_data": self.model_data,
            "model_uid": self.model_uid,
        }

    def _is_endpoint_deployed(self) -> bool:
        """
        assert if model endpoint is deployed in WML deployment space
        :param
        :return: Boolean True if model in deployed False if it is not deployed
        """
        if self.model_endpoint:
            wml_client.set.default_space(WML_SPACE_ID)
            is_deploy = bool(
                get_deployment_uid_by_name(
                    deployment_name=self.model_endpoint, wml_client=wml_client
                )
            )
            self.is_deployed = is_deploy
        else:
            is_deploy = False
        return is_deploy

    def _assign_params_from_Endpoint(self, ep_name: str):
        deployment_uid = get_deployment_uid_by_name(
            deployment_name=ep_name, wml_client=wml_client
        )
        deployment_details = wml_client.deployments.get_details(
            deployment_uid=deployment_uid
        )
        self.model_uid = deployment_details.get("entity").get("asset").get("id")
        self.model_endpoint = ep_name
        self.is_deployed = True

    def _assign_params_from_WOS(self, subscription_id: str):
        self.subscription_id = subscription_id
        self.is_monitored = True

    def _assign_params_from_model(self):
        if self.model_uid:
            model_details = wml_client.repository.get_model_details(
                model_uid=self.model_uid
            )
            self._model_config.class_label = model_details
            self._model_config.inference_instance = (
                model_details.get("entity").get("software_spec").get("name")
            )
            runtime, version = model_details.get("entity").get("type").split("_")
            self._model_config.inference_framework = runtime
            self._model_config.inference_framework_version = version
            self._model_config.class_label = model_details.get("entity").get(
                "label_column"
            )
            if training_data_references := model_details.get("entity").get(
                "training_data_references"
            ):
                self._model_config.feature_columns = [
                    field.get("name")
                    for field in training_data_references[0].get("schema").get("fields")
                ]
            self.model_data = self.model_uid
            self.is_trained = True
            self.is_governed = True

    def _assign_model_endpoint(self) -> str:
        return f"wml_{self._model_config.source_dir}_{time.strftime('%Y_%m_%d_%H_%M', time.gmtime())}-{randint(100, 999)}"

    def _reset_states(self):
        self.model_data = None
        self.model_uid = None
