# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Union
import json
import os
import time
import importlib

from mlmonitor.src import (
    API_KEY,
    CATALOG_ID,
    MODEL_ENTRY_ID,
    logger,
    DATA_ROOT,
    MODEL_ROOT,
    ENV,
    PROJECT_ROOT,
    AUTH_ENDPOINT,
    USERNAME,
    VERIFY_CP4D_SSL,
)

from mlmonitor.src.model.config_azure import AzureModelConfig
from mlmonitor.src.model.use_case import ModelUseCase

from mlmonitor.src.azure import AZ_WORKSPACE
from azureml.core.model import Model
from mlmonitor.src.azure.deployment import (
    get_deploy_facts,
    is_deployed,
    register_model,
    get_deployments,
    deploy_az_model,
    get_model_uid_by_name,
)
from mlmonitor.src.azure.train import train_az_ml_job
from mlmonitor.src.azure.scoring import get_wos_response, _score_unstructured
from mlmonitor.src.wos.run_payload_logging import log_as_wos_payload
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.wos.evaluate import evaluate_monitor
from mlmonitor.src.demos.scenario_helpers import perturb_column, perturb_double_column
from mlmonitor.src.factsheets.utils import (
    get_model_id_by_deployment_name,
    FactsheetHelpers,
)


class AzureModelUseCaseEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, AzureModelUseCase):
            return obj.json_object()
        return super().default(obj)


class AzureModelUseCase(ModelUseCase):
    """
    ModelUseCase trained and deployed in Azure Machine Learning
    governed in CloudPak for Data AI FactSheets
    monitored in Watson OpenScale.

    _model_config class attribute from is a AzureModelConfig Object that contain all the relevant parameters
    to bring a AzureModelUseCase instance to production (monitor state)

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

        self.load_azure_use_case(file=file)

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
    def job_name(self) -> str:
        """Config property"""
        return self._job_name

    @job_name.setter
    def job_name(self, value: str):
        """
        Set the job_name that produce Azure model
        :param value:
        :return:
        """
        self._job_name = value

    @property
    def ibm_key_name(self) -> str:
        """ ""Model Use Case ibm_key_name property for Azure keyvault"""
        return self._ibm_key_name

    @ibm_key_name.setter
    def ibm_key_name(self, value: str):
        """
        Set the ibm_key_name to use by Azure keyvault
        :param value:
        :return:
        """
        self._ibm_key_name = value

    @property
    def model_uid(self) -> str:
        """Model ID once an Azure model is published as an Asset in a Deployment Space"""
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
        Save AzureModelUseCase to JSON file to AzureModelUseCase Object
        :param file: json file path where serialized AzureModelUseCase Object should be saved
        :return:
        """
        if not file:
            file = os.path.join("./", "AzureModelUseCase.json")
        logger.info(f"AzureModelUseCase class saved to {file}")
        with open(file, "w") as f:
            json.dump(obj=self, fp=f, cls=AzureModelUseCaseEncoder, indent=4)

    def load_azure_use_case(self, file: Union[str, Dict] = None):
        """
        Instantiate AzureModelUseCase Object from a file or a python dictionary

        :param self: Reference the class instance
        :param file:Union[str,Dict] Specify the file or Dictionary to load
        :return:
        """

        if file is None:
            loaded_dict = {}
            logger.info("AzureModelUseCase class initialized from scratch")
        elif isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
                logger.info(f"AzureModelUseCase class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logger.info("AzureModelConfig class loaded from dict")
        else:
            raise ValueError("should be a file path or dictionary")

        self.serving_platform = "azure"
        self._df = None
        self.model_data = loaded_dict.get("model_data")
        self.model_uid = loaded_dict.get("model_uid")
        self.job_name = loaded_dict.get("job_name")
        self._model_config = AzureModelConfig(
            source_dir=self.source_dir,
            file=loaded_dict.get("model_config", None),  # = model_signature.json
        )
        # assign a model endpoint name
        self.model_endpoint = loaded_dict.get(
            "model_endpoint", self._assign_model_endpoint()
        )

        self.ibm_key_name = loaded_dict.get("ibm_key_name", "IBM-API-KEY-MLOPS")

    def json_object(self):
        """
        returns a json object of AzureModelUseCase class that can be used for serialization and deserialization of the Object.
        :param self: Access the class attributes
        :return: A dictionary is used by AzureModelUseCaseEncoder. This dictionary is compatible to load a AzureModelUseCase Object
        """
        BaseUseCaseObject = self.base_use_case_json()
        BaseUseCaseObject["__class__"] = "AzureModelUseCase"
        # update model config related to Azure
        AzureModelUseCaseObject = self.azure_use_case_json()

        AzureModelConfigObject = self._model_config.json_object()

        return {
            **BaseUseCaseObject,
            **AzureModelUseCaseObject,
            **{"model_config": AzureModelConfigObject},
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
        # CHECK Azure Deployments

        if self._is_endpoint_deployed():
            self._assign_params_from_Endpoint(ep_name=endpoint_name)
        else:
            self.model_uid = get_model_uid_by_name(
                model_name=self.model_endpoint, workspace=AZ_WORKSPACE, key="id"
            )
        # TODO investigate why AIGOv client disables mlmonitor logs
        # Check for AI FactSheets
        # fs_client = self._init_external_fs_client()
        # all_models = fs_client.assets.get_ai_usecase(
        #     ai_usecase_id=self.model_entry_id, catalog_id=self.catalog_id
        # ).get_all_facts().get('entity').get('modelfacts_global')

        fs_helpers = FactsheetHelpers(
            api_key=API_KEY,
            container_type="catalog",
            container_id=self.catalog_id,
            model_entry_id=self.model_entry_id,
            env=ENV,
            username=USERNAME,
            cpd_url=AUTH_ENDPOINT,
        )

        all_models = fs_helpers.get_models(verify=VERIFY_CP4D_SSL)

        if all_models.get("physical_models"):
            if asset_id := get_model_id_by_deployment_name(
                models=all_models, deployment_name=endpoint_name
            ):
                self._assign_params_from_Factsheets(ep_name=endpoint_name)
                logger.info(
                    f"Model Endpoint [{self.model_endpoint}] is Governed in AI Factsheets with asset [{asset_id}]"
                )
            else:
                logger.info(f"No Governance found for [{endpoint_name}]")

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

    def train(self):
        """
        train Azure model based on model use case definition for inference (runtime,script...) in azure_runtime section of model signature
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, job name and state of training
        """

        self._store_ibm_key()
        if not self.model_endpoint:
            model_name = self._assign_model_endpoint()
        else:
            model_name = self.model_endpoint

        model = train_az_ml_job(
            model_name=model_name,
            model_config=self._model_config,
            workspace=AZ_WORKSPACE,
            model_entry_id=self.model_entry_id,
            catalog_id=self.catalog_id,
            data_path=DATA_ROOT,
            ibm_key_name=self.ibm_key_name,
            cp4d_env=ENV,
            cp4d_username=USERNAME,
            cp4d_url=AUTH_ENDPOINT,
        )

        self.model_uid = model.id
        if not self.model_endpoint:
            self.model_endpoint = model_name
        self._assign_params_from_model()

    def _init_external_fs_client(self):
        """Instantiate External Factsheets Client

        :return: AI Factsheets client
        """

        from ibm_aigov_facts_client import CloudPakforDataConfig, AIGovFactsClient

        aigov_params = {
            "experiment_name": self.model_endpoint,
            "set_as_current_experiment": True,
            "enable_autolog": True,
            "external_model": True,
        }

        if ENV == "prem":
            from mlmonitor.src import USERNAME

            aigov_params["cloud_pak_for_data_configs"] = CloudPakforDataConfig(
                service_url=AUTH_ENDPOINT, username=USERNAME, api_key=API_KEY
            )
        elif ENV == "saas":
            aigov_params["api_key"] = API_KEY

        return AIGovFactsClient(**aigov_params)

    def _train_local(self):
        """
        train Azure model in local environment based on model use case definition for inference (runtime,script...)
        in azure_runtime section of model signature
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, job name and state of training
        """
        # Specifying endpoints if not already assigned
        if not self.model_endpoint:
            self.model_endpoint = self._assign_model_endpoint()

        train = getattr(
            importlib.import_module(
                f"mlmonitor.{self._model_config.source_dir}.{self._model_config.train_module}"
            ),
            self._model_config.train_method,
        )

        facts_client = self._init_external_fs_client()
        parameters = self._model_config.hyperparameters

        # Dictionary of metrics, dict of params, dict of tags
        local_model_data = train(
            model_dir=MODEL_ROOT,
            data_path=os.path.join(DATA_ROOT, self._model_config.data_dir),
            train_dataset=self._model_config.training_data,
            val_dataset=self._model_config.validation_data,
            test_dataset=self._model_config.test_data,
            logger=logger,
            **parameters,
        )

        fs_model = facts_client.external_model_facts.save_external_model_asset(
            model_identifier=self.model_endpoint,
            name=self.model_endpoint,
            schemas=None,  # TODO add schemas
            training_data_reference=None,  # TODO add training_data_reference
            description=f"mlmonitor {self.source_dir}",
            catalog_id=self.catalog_id,
        )

        muc_utilities = facts_client.assets.get_ai_usecase(
            ai_usecase_id=self.model_entry_id,
            catalog_id=self.catalog_id,
        )

        fs_model.track(
            usecase=muc_utilities,
            approach=muc_utilities.get_approaches()[0],
            version_number="minor",  # "0.1.0"
        )

        # Publish model to Azure
        az_model = register_model(
            workspace=AZ_WORKSPACE,
            model_path=local_model_data,
            model_name=self.model_endpoint,
            tags={
                "use_case": self.source_dir,
                "factsheets_url": f"{AUTH_ENDPOINT}/data/catalogs/{self.catalog_id}/asset/{self.model_entry_id}/asset-preview",
            },
            description=f"trained by mlmonitor on {time.strftime('%Y-%m-%d')}",
        )

        exp = fs_model.get_experiment(self.model_endpoint)
        run = exp.get_run()

        if len(az_model.tags) > 0:
            run.set_custom_tags(tags_dict=az_model.tags)

        params_dict = {
            "az_model_version": str(az_model.version),
            "az_model_url": str(az_model.url),
            "az_model_id": str(az_model.id),
            "az_model_name": str(az_model.name),
        }
        print(params_dict)
        run.set_custom_params(params_dict=params_dict)

        self.model_uid = az_model.id
        self.model_data = az_model.url
        self.is_trained = True

    def deploy(self):
        """
        deploy Azure model based on model use case definition for inference (runtime,script...).
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, deployment details
        """
        if self.model_data:
            deploy_config_params = {
                "cpu_cores": self._model_config.cpu_cores,
                "memory_gb": self._model_config.memory_gb,
                "tags": self._model_config.tags,
                "description": self._model_config.description,
            }

            model = Model(workspace=AZ_WORKSPACE, id=self.model_uid)
            if not self._is_endpoint_deployed():
                deploy_az_model(
                    compute_type=self._model_config.inference_compute,
                    workspace=AZ_WORKSPACE,
                    model=model,
                    cluster_name=self._model_config.aks_cluster_name,
                    entry_script=os.path.join(
                        PROJECT_ROOT,
                        self.source_dir,
                        self._model_config.inference_script,
                    ),
                    auth_enabled=self._model_config.auth_enabled,  # authentication not working ACI and WOS
                    environ_name=self.source_dir,
                    deployment_name=self.model_endpoint,
                    conda_packages=self._model_config.conda_packages,
                    redeploy=False,
                    deploy_config_params=deploy_config_params,
                )
            else:
                logger.warning(
                    f"Azure model endpoint {self.model_endpoint} already deployed"
                )
            self.is_deployed = True
        else:
            logger.error(
                "Model Endpoint cannot be deployed without valid model data location `model_data` attribute"
            )

    def govern(self):
        """
        not needed for Azure models
        """
        from ibm_aigov_facts_client import DeploymentDetails

        if not self.model_data:
            raise ValueError("Model data location should be set")

        if not self.model_endpoint:
            raise ValueError("Model endpoint name should be set")

        facts_client = self._init_external_fs_client()

        deployment = DeploymentDetails(
            identifier=self.model_endpoint,
            name=self.model_endpoint,
            deployment_type="online",
            scoring_endpoint=self.model_endpoint,
        )

        fs_model = facts_client.external_model_facts.save_external_model_asset(
            model_identifier=self.model_endpoint,
            name=self.model_endpoint,
            deployment_details=deployment,
            catalog_id=self.catalog_id,
        )

        exp = fs_model.get_experiment(self.model_endpoint)
        run = exp.get_run()

        deploy_metrics, deployment_params, deploy_tags = get_deploy_facts(
            workspace=AZ_WORKSPACE, deployment_name=self.model_endpoint
        )

        run.set_custom_metrics(metrics_dict=deploy_metrics)
        run.set_custom_tags(tags_dict=deploy_tags)
        run.set_custom_params(params_dict=deployment_params)

        self.is_governed = True

    def score_model(self):
        """
        Perform scoring request to deployed Azure model (structured or unstructured datasets)
        :param
        :return: model predictions
        """
        if not self._is_endpoint_deployed():
            logger.warning(f"Azure model endpoint {self.model_endpoint} not deployed")
            return {"fields": {}, "values": {}}

        inference_samples = 2
        test_data = self._model_config._get_data(
            dataset_type="test", num_samples=inference_samples
        )

        if self._model_config.data_type == "structured":

            df = test_data.loc[:, self._model_config.feature_columns]

            response_wos = get_wos_response(
                df=df, endpoint_name=self.model_endpoint, workspace=AZ_WORKSPACE
            )

        elif self._model_config.data_type == "unstructured_image":

            samples, labels = test_data
            print("samples shape sent for inference", samples.shape)

            result = _score_unstructured(
                payload=samples,
                endpoint_name=self.model_endpoint,
                workspace=AZ_WORKSPACE,
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
            f"Azure Endpoint {self.model_endpoint} scoring response:\n{response_wos}"
        )

        return response_wos

    @log_as_wos_payload("azure")
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

    def azure_use_case_json(self) -> Dict:
        """
        returns a dictionary containing the job_name, catalog_id, and model_endpoint of a SageMaker Model Use case.

        :param self: Access the class attributes
        :return: dictionary with job_name, catalog_id and model_endpoint as keys
        """
        return {
            "catalog_id": self.catalog_id,
            "model_data": self.model_data,
            "model_uid": self.model_uid,
            "job_name": self.job_name,
            "ibm_key_name": self.ibm_key_name,
        }

    def _is_endpoint_deployed(self) -> bool:
        """
        assert if model endpoint is deployed in Azure deployment space
        :param
        :return: Boolean True if model in deployed False if it is not deployed
        """

        self.is_deployed = is_deployed(
            deployment_name=self.model_endpoint, workspace=AZ_WORKSPACE
        )

        return self.is_deployed

    def _assign_params_from_Endpoint(self, ep_name: str):
        deployments = get_deployments(
            deployment_name=self.model_endpoint, workspace=AZ_WORKSPACE
        )
        deployment_details = deployments[0]
        self._model_config.inference_compute = deployment_details.compute_type.lower()
        self._model_config.conda_packages = (
            list(deployment_details.environment.python.conda_dependencies.pip_packages)
            if deployment_details.environment
            else None
        )
        self.model_endpoint = ep_name
        self.is_deployed = True
        if len(deployment_details.models) == 1:
            model = deployment_details.models[0]
            self.model_uid = model.id
        elif len(deployment_details.models) >= 1:
            logger.warning(
                f"{len(deployment_details.models)} models found for endpoint {self.model_endpoint}"
            )
        else:
            logger.error(f"No model found for endpoint {self.model_endpoint}")

    def _assign_params_from_WOS(self, subscription_id: str):
        self.subscription_id = subscription_id
        self.is_monitored = True

    def _assign_params_from_Factsheets(self, ep_name: str):
        self.is_governed = True

    def _assign_params_from_model(self):
        if self.model_uid:
            model = Model(workspace=AZ_WORKSPACE, id=self.model_uid)
            self.model_data = model.url
            self.is_trained = True
            if model.run:
                self.job_name = model.run.id

    def _assign_model_endpoint(self) -> str:
        return f"az-{self._model_config.source_dir}-{time.strftime('%m-%d-%H-%M', time.gmtime())}".replace(
            "_", "-"
        )

    def _reset_states(self):
        self.model_data = None
        self.model_uid = None
        self.job_name = None

    def _store_ibm_key(self):
        keyvault = AZ_WORKSPACE.get_default_keyvault()
        keyvault.set_secret(name=self.ibm_key_name, value=API_KEY)
