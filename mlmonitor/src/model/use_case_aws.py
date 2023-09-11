# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Union
import json
import os
import boto3
from boto3.session import Session as boto_Session
import warnings

from mlmonitor.src import (
    ENV,
    DATA_ROOT,
    logger,
    USERNAME,
    AUTH_ENDPOINT,
    API_KEY,
    CATALOG_ID,
    MODEL_ENTRY_ID,
    aws_credentials,
    VERIFY_CP4D_SSL,
)

from mlmonitor.src.model.config_aws import SageMakerModelConfig
from mlmonitor.src.model.use_case import ModelUseCase
from mlmonitor.src.wos.evaluate import evaluate_monitor
from mlmonitor.src.aws.train_sagemaker_job import train_sagemaker_job
from mlmonitor.src.aws.deploy_sagemaker_endpoint import (
    deploy_sagemaker_endpoint,
    govern_sagemaker_endpoint,
)
from mlmonitor.src.wos.run_payload_logging import log_as_wos_payload
from mlmonitor.src.wos import wos_client
from mlmonitor.src.aws.score_sagemaker_ep import score_sagemaker_endpoint
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.aws.training import is_training_job_completed
from mlmonitor.src.aws.deployment import is_deployed
from mlmonitor.src.aws.secrets_manager import (
    sm_create,
    sm_update,
    sm_secret_name_exists,
    sm_secret_key_name_exists,
)
from mlmonitor.src.aws.scoring import sm_get_modelnames  # , sm_get_ep_configname
from mlmonitor.src.factsheets.utils import (
    get_model_id_by_deployment_name,
    FactsheetHelpers,
)
from mlmonitor.exceptions import MisMatchKeyManagerSecretValue
from mlmonitor.src.demos.scenario_helpers import perturb_column, perturb_double_column


class SageMakerModelUseCaseEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, SageMakerModelUseCase):
            return obj.json_object()
        return super().default(obj)


class SageMakerModelUseCase(ModelUseCase):
    """
    ModelUseCase trained and deployed in AWS SageMaker
    governed in CloudPak for Data AI FactSheets
    monitored in Watson OpenScale.

    _model_config class attribute from is a SageMakerModelConfig Object that contain all the relevant parameters
    to bring a SageMakerModelUseCase instance to production (monitor state)

    supported states are :

        - "trained"
        - "deployed"
        - "governed"
        - "monitored"

    """

    sm_secret_name = "IBM_KEYS"

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

        self.load_aws_use_case(file=file)

    @property
    def ibm_key_name(self) -> str:
        """ ""Model Use Case ibm_key_name property for Sagemaker secret manager"""
        return self._ibm_key_name

    @ibm_key_name.setter
    def ibm_key_name(self, value: str):
        """
        Set the ibm_key_name to use in Sagemaker manager
        :param value:
        :return:
        """

        value = value.strip("") if value else value
        client = self._get_secrets_manager()

        exists, match = sm_secret_key_name_exists(
            client=client,
            secret_name=self.sm_secret_name,
            secret_value=API_KEY,
            secret_key_name=value,
            logger=logger,
        )

        if exists:
            if not match:
                warnings.warn(
                    MisMatchKeyManagerSecretValue.mismatch_keymanager_secret_value(
                        value, API_KEY
                    ).message
                )
            else:
                logger.info(
                    f"secret name {self.sm_secret_name} already exists with key name {value}"
                )
        elif not sm_secret_name_exists(
            client=client, secret_name=self.sm_secret_name, logger=logger
        ):
            sm_create(
                client=client,
                secret_name=self.sm_secret_name,
                secret_key_name=value,
                secret_key_value=API_KEY,
                logger=logger,
            )
        else:
            logger.info(f"secret name {self.sm_secret_name} exists")
            sm_update(
                client=client,
                secret_name=self.sm_secret_name,
                secret_key_name=value,
                secret_key_value=API_KEY,
                logger=logger,
            )

        logger.info(f"set ibm_key_name to {value}")
        self._ibm_key_name = value

    @property
    def model_data(self) -> str:
        """Model Use Case property for model binary location in S3 bucket"""
        return self._model_data

    @model_data.setter
    def model_data(self, value: str):
        """
        Set the model_data to use
        :param value:
        :return:
        """
        self._model_data = value

    @property
    def job_name(self) -> str:
        """ ""Model Use Case job_name property for Sagemaker job name triggered to train this model"""
        return self._job_name

    @job_name.setter
    def job_name(self, value: str):
        """
        Set the job_name to use
        :param value:
        :return:
        """
        self._job_name = value

    def save_use_case(self, file: str = None):
        """
        Save SageMakerModelUseCase to JSON file to SageMakerModelUseCase Object
        :param file: json file path where serialized SageMakerModelUseCase Object should be saved
        :return:
        """
        if not file:
            file = os.path.join("./", "SageMakerModelUseCase.json")
        logger.info(f"SageMakerModelUseCase class saved to {file}")
        with open(file, "w") as f:
            json.dump(obj=self, fp=f, cls=SageMakerModelUseCaseEncoder, indent=4)

    def load_aws_use_case(self, file: Union[str, Dict] = None):
        """
        Instantiate SageMakerModelUseCase Object from a file or a python dictionary

        :param self: Reference the class instance
        :param file:Union[str,Dict] Specify the file or Dictionary to load
        :return:
        """

        if file is None:
            loaded_dict = {}
            logger.info("SageMakerModelUseCase class initialized from scratch")
        elif isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
                logger.info(f"SageMakerModelUseCase class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logger.info("SageMakerModelUseCase class loaded from dict")
        else:
            raise ValueError("should be a file path or dictionary")

        self.serving_platform = "aws"
        self.model_data = loaded_dict.get("model_data")
        self.job_name = loaded_dict.get("job_name")
        self._model_config = SageMakerModelConfig(
            source_dir=self.source_dir, file=loaded_dict.get("model_config", None)
        )
        self.ibm_key_name = loaded_dict.get("ibm_key_name", "IBM_API_KEY_MLOPS")
        self._df = None

    def json_object(self):
        """
        returns a json object of SageMakerModelUseCase class that can be used for serialization and deserialization of the Object.
        :param self: Access the class attributes
        :return: A dictionary is used by SageMakerModelUseCaseEncoder. This dictionary is compatible to load a SageMakerModelUseCase Object
        """
        BaseUseCaseObject = self.base_use_case_json()
        BaseUseCaseObject["__class__"] = "SageMakerModelUseCase"
        # update model config related to Sagemaker
        SageMakerModelUseCaseObject = self.sm_use_case_json()

        SageMakerModelConfigObject = self._model_config.json_object()

        return {
            **BaseUseCaseObject,
            **SageMakerModelUseCaseObject,
            **{"model_config": SageMakerModelConfigObject},
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
        # By Convention all Endpoint Names should be Matched to their respective Job Names
        self.job_name = endpoint_name

        self.is_trained = False
        self.is_deployed = False
        self.is_governed = False
        self.is_monitored = False

        # CHECK FOR DEPLOYMENTS IN AWS Sagemaker
        if self._is_endpoint_deployed():
            logger.info(f"endpoint_name {endpoint_name} deployed")
            self._assign_params_from_Endpoint(ep_name=endpoint_name)
        else:
            logger.info(f"No Endpoint Deployed found for [{endpoint_name}]")
        # CHECK FOR TRAINING JOBS IN AWS Sagemaker
        if is_training_job_completed(job_name=self.job_name):
            logger.info(
                f"Training Job  [{self.job_name}] exists for endpoint_name [{endpoint_name}]"
            )
            self._assign_params_from_Job(job_name=self.job_name)
        else:
            logger.info(f"No Training Job found for [{self.job_name}]")
        # CHECK FOR Assets in AI Factsheets
        fs_helpers = FactsheetHelpers(
            api_key=API_KEY,
            container_type="catalog",
            container_id=self.catalog_id,
            model_entry_id=self.model_entry_id,
            env=ENV,
            username=USERNAME,
            cpd_url=AUTH_ENDPOINT,
        )

        models = fs_helpers.get_models(verify=VERIFY_CP4D_SSL)

        # With AI Factsheets and AWS SageMaker integration
        #
        # model asset id => is always mapped to Sagemaker Model Name (
        # named randomly assigned in SM) e.g sagemaker-scikit-learn-2023-01-25-19-33-06-574
        #
        # deployment asset id => corresponds to Sagemaker Endpoint Name (name can be assigned by user) e.g
        # sm-cc-xgboost-2023-01-25-19-28-31-145 In this Projects jobname are always assigned as Sagemaker Endpoint
        # Name during deployments
        #
        # [{'id': 'a9abc6b4-9cd1-4d9e-90c4-9b23d9ee558a', 'name': 'sagemaker-scikit-learn-2023-01-25-19-33-06-574',
        # 'type': 'external_model', 'container_id':
        # '2885333d-637c-49a0-a9ca-23323377a36a', 'container_name': 'Development space', 'container_type': 'catalog',
        # 'is_deleted': False, 'master_id': '80d226ec64aaad0787b91608cbe89f9e', 'source_asset_id': '', 'algorithm':
        # '', 'deployment_space_type': 'development', 'deployments': [{'id': '80d226ec64aaad0787b91608cbe89f9e',
        # 'status': 'Pending Evaluation', 'name': 'sm-cc-xgboost-2023-01-25-19-28-31-145'}], 'openpages_details': {}}]

        if models.get("physical_models"):
            if asset_id := get_model_id_by_deployment_name(
                models=models, deployment_name=endpoint_name
            ):
                self._assign_params_from_Factsheets(ep_name=endpoint_name)
                logger.info(
                    f"Model Endpoint [{self.model_endpoint}] is Governed in AI Factsheets with asset [{asset_id}]"
                )
            else:
                logger.info(f"No Governance found for [{endpoint_name}]")

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
                self._assign_params_from_WOS(subscription_id=subscription_ids[0])

    def _train_local(self):
        """
        train model in local environment
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, job name and state of training
        """
        pass

    def train(self):
        """
        train Sagemaker model based on model use case definition for inference (runtime,script...).
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, job name and state of training
        """
        model_data = train_sagemaker_job(
            model_config=self._model_config,
            custom=True,
            ibm_key_name=self.ibm_key_name,
            model_entry_id=self.model_entry_id,
            catalog_id=self.catalog_id,
            data_path=os.path.join(DATA_ROOT, self._model_config.data_dir),
            cp4d_env=ENV,
            cp4d_username=USERNAME,
            cp4d_url=AUTH_ENDPOINT,
        )

        self.model_data = model_data.get("model_data")
        self.job_name = model_data.get("job_name")
        self.is_trained = True

    def deploy(self):
        """
        deploy Sagemaker model based on model use case definition for inference (runtime,script...).
        :param self: Access fields or methods of the class in python
        :return: A dictionary with the model data, deployment details
        """

        self.model_endpoint = self.job_name

        if not self.job_name:
            raise ValueError(
                f"No training job (job_name parameter) for this {self.source_dir} use case. this use case should be in `trained` state and associated with a Completed training job "
            )

        if not is_training_job_completed(job_name=self.job_name):
            self.is_trained = False
            raise ValueError(
                f"{self.job_name} training job not Completed or does not exists"
            )

        if not self._is_endpoint_deployed():
            deploy_sagemaker_endpoint(
                model_config=self._model_config,
                deployment_name=self.job_name,
                model_data=self.model_data,
                ibm_key_name=self.ibm_key_name,
            )
        else:
            logger.warning(f"Endpoint {self.job_name} Already Deployed")

        self.is_deployed = True

    def govern(self):
        """
        It takes an instance of a ModelConfig class, and uses it to create or update
        the associated AI Factsheets asset with deployment details.

        :param self: Access fields or methods of the class in python
        :return:
        """
        if self._is_endpoint_deployed():
            govern_sagemaker_endpoint(
                model_config=self._model_config,
                deployment_name=self.model_endpoint,
                catalog_id=self.catalog_id,
                model_entry_id=self.model_entry_id,
                env=ENV,
                cp4d_url=AUTH_ENDPOINT,
                cp4d_username=USERNAME,
            )

            self.is_governed = True
        else:
            logger.warning(
                f"Endpoint {self.model_endpoint} Should be deployed before sending deployments details to AI Factsheets"
            )

    def score_model(self):
        """
        Perform scoring request to deployed Sagemaker model (structured or unstructured datasets)
        :param
        :return: model predictions
        """

        if not self.model_endpoint:
            raise ValueError("score_model No Model Endpoint assigned to this Object")

        if not self._is_endpoint_deployed():
            self.is_deployed = False
            raise ValueError(
                f"score_model No Model Endpoint deployed with name : {self.model_endpoint}"
            )

        score_parameters = {
            "deployment_name": self.model_endpoint,
            "inference_samples": 10,
            "aws_credentials": aws_credentials,
            "model_config": self._model_config,
        }

        response = score_sagemaker_endpoint(**score_parameters)
        logger.debug(f"score_model sagemaker response:\n{response}")
        return response

    @log_as_wos_payload("aws")
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

    def sm_use_case_json(self) -> Dict:
        """
        returns a dictionary containing the job_name, catalog_id, and model_endpoint of a SageMaker Model Use case.

        :param self: Access the class attributes
        :return: dictionary with job_name, catalog_id and model_endpoint as keys
        """
        return {
            "job_name": self.job_name,
            "catalog_id": self.catalog_id,
            "ibm_key_name": self.ibm_key_name,
        }

    def _is_endpoint_deployed(self) -> bool:
        """
        assert if model endpoint is deployed in AWS Sagemaker
        :param
        :return: Boolean True if model in deployed False if it is not deployed
        """
        return is_deployed(deployment_name=self.model_endpoint)

    def _assign_params_from_Job(self, job_name: str):

        session = boto_Session()
        sm_client = session.client("sagemaker")

        job_details = sm_client.describe_training_job(TrainingJobName=job_name)
        self.model_data = job_details.get("ModelArtifacts").get("S3ModelArtifacts")
        self.catalog_id = job_details.get("HyperParameters").get("catalog-id")
        self.model_entry_id = job_details.get("HyperParameters").get("model-entry-id")
        self.ibm_key_name = job_details.get("HyperParameters").get("ibm-key-name")
        self._model_config.train_script = job_details.get("HyperParameters").get(
            "sagemaker_program"
        )
        self._model_config.hyperparameters = job_details.get("HyperParameters")
        self._model_config.train_instance = (
            job_details.get("ResourceConfig").get("InstanceType").strip('"')
        )
        self.is_trained = True

    def _assign_params_from_Endpoint(self, ep_name: str):

        session = boto_Session()
        sm_client = session.client("sagemaker")
        self.model_endpoint = ep_name
        self.is_deployed = True

        # endpoint_cfg_name = sm_get_ep_configname(
        #    sagemaker_client=sm_client, endpoint_name=ep_name
        # )

        # endpoint_cfg_description = sm_client.describe_endpoint_config(
        #    EndpointConfigName=endpoint_cfg_name
        # )

        model_names = sm_get_modelnames(
            sagemaker_client=sm_client, endpoint_name=ep_name
        )
        model_name = model_names[0]
        model_description = sm_client.describe_model(ModelName=model_name)

        self._model_config.inference_script = (
            model_description.get("PrimaryContainer")
            .get("Environment")
            .get(
                "SAGEMAKER_PROGRAM", "not_found.py"
            )  # works only for Scikit containers
        )
        self.model_data = model_description.get("PrimaryContainer").get("ModelDataUrl")
        # from mlmonitor.src.utils.utils import jobname_from_modeldata
        # self.job_name = jobname_from_modeldata(self.model_data)
        self.job_name = ep_name

    def _assign_params_from_WOS(self, subscription_id: str):
        self.subscription_id = subscription_id
        self.is_monitored = True

    def _assign_params_from_Factsheets(self, ep_name: str):
        self.is_governed = True

    def _get_secrets_manager(self):
        session = boto3.session.Session()
        return session.client(service_name="secretsmanager", **aws_credentials)

    def replace_secrets_manager(self):
        sm_client = self._get_secrets_manager()
        sm_update(
            client=sm_client,
            secret_name=self.sm_secret_name,
            secret_key_name=self.ibm_key_name,
            secret_key_value=API_KEY,
            logger=logger,
        )

    def _reset_states(self):
        self.model_data = None
        self.job_name = None
