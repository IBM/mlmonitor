# SPDX-License-Identifier: Apache-2.0
import json
import os
import time
from ibm_aigov_facts_client import (
    AIGovFactsClient,
    ModelEntryProps,
    ExternalModelSchemas,
    CloudPakforDataConfig,
    TrainingDataReference,
)

try:
    from mlmonitor.use_case_churn.utils import git_branch
except ImportError:
    from utils import git_branch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def init_external_fs_client(logger, **params) -> tuple:
    """
    initializes the AIGovFactsClient object for AI Factseets for external model and returns a tuple of instantiated objects.

     fs_tags : Factsheets tags specific to Sagemaker container where training job is running
     fs_params : Factsheets parameters specific to Sagemaker container where training job is running

     if training running locally (not in SM job), only git branch is set as tag

    :param logger: logger passed to the function
    :param **params: Pass in parameters to instantiate AIGovFactsClient (catalog_id,model_entry_id,ibm_key_name,region_name)
    :return: A tuple containing the following: (facts_client, props, experiment_name, experiment_id , fs_tags, fs_params)
    """

    API_KEY_NAME = params.get("ibm_key_name")
    region_name = params.get("region_name", "ca-central-1")
    fs_params = {}

    if os.environ.get("SM_TRAINING_ENV"):
        from utils import _get_secret

        SM_TRAINING_ENV = json.loads(os.environ["SM_TRAINING_ENV"])
        experiment_name = SM_TRAINING_ENV["job_name"]

        API_KEY = json.loads(
            _get_secret(secret_name="IBM_KEYS", region_name=region_name)
        ).get(API_KEY_NAME)
        fs_tags = {
            "job_name": SM_TRAINING_ENV["job_name"],
            "module_dir": SM_TRAINING_ENV.get("module_dir"),
            "train": SM_TRAINING_ENV.get("channel_input_dirs").get("train"),
            "validation": SM_TRAINING_ENV.get("channel_input_dirs").get("validation"),
            "user_entry_point": SM_TRAINING_ENV["user_entry_point"],
        }

        fs_params = {
            "num_cpus": SM_TRAINING_ENV.get("num_cpus"),
            "num_gpus": SM_TRAINING_ENV.get("num_gpus"),
        }

        logger.info(f"Running in a SM JOB {experiment_name}")
    else:
        API_KEY = os.getenv("API_KEY")
        time_suffix = time.strftime("%Y%m%d-%H%M", time.gmtime())
        experiment_name = f"sm-cc-xgboost-{time_suffix}"
        fs_tags = {"git_branch": git_branch(logger=logger, path=PROJECT_ROOT)}

    # Init Factsheets Client
    if params.get("catalog_id") and params.get("model_entry_id"):
        props = ModelEntryProps(
            model_entry_catalog_id=params.get("catalog_id"),
            model_entry_id=params.get("model_entry_id"),
        )
    else:
        props = None

    if params.get("cp4d_env") == "saas":
        facts_client = AIGovFactsClient(
            api_key=API_KEY,
            experiment_name=experiment_name,
            external_model=True,
            enable_autolog=True,
            set_as_current_experiment=True,
        )
    elif params.get("cp4d_env") == "prem":
        facts_client = AIGovFactsClient(
            cloud_pak_for_data_configs=CloudPakforDataConfig(
                service_url=params.get("cp4d_url"),
                username=params.get("cp4d_username"),
                api_key=API_KEY,
            ),
            experiment_name=experiment_name,
            external_model=True,
            enable_autolog=True,
            set_as_current_experiment=True,
        )

    # get experiment_id
    experiment_id = facts_client.experiments.get_current_experiment_id()
    logger.info(f"Current Experiment {experiment_name} ID {experiment_id}")

    return facts_client, props, experiment_name, experiment_id, fs_tags, fs_params


def save_fs_model(
    logger,
    facts_client: AIGovFactsClient,
    experiment_id: str,
    experiment_name: str,
    catalog_id: str,
    model_entry_id: str,
    inputs=None,
    outputs=None,
    tdataref=None,
    tags: dict = {},
    params: dict = {},
    metrics: dict = {},
):
    external_schemas = None
    runs = facts_client.runs.list_runs_by_experiment(experiment_id)
    logger.info(f"runs : {runs}")
    run_id = facts_client.runs.get_current_run_id()
    logger.info(f"Current RunID {run_id}")  # TODO investigate None run_id on azure job

    if len(metrics) > 0:
        facts_client.runs.log_metrics(run_id, metrics)
        logger.info(f"save_fs_model FactSheets metrics :\n{metrics}")
    if len(tags) > 0:
        facts_client.runs.set_tags(run_id, tags)
        logger.debug(f"save_fs_model FactSheets tags :\n{json.dumps(tags, indent=4)}")
    if len(params) > 0:
        facts_client.runs.log_params(run_id, params)
        logger.debug(
            f"save_fs_model FactSheets params :\n{json.dumps(params, indent=4)}"
        )

    if inputs and outputs:
        external_schemas = ExternalModelSchemas(input=inputs, output=outputs)

    trainingdataref = TrainingDataReference(schema=tdataref) if tdataref else None
    facts_client.export_facts.export_payload(run_id)
    fs_model = facts_client.external_model_facts.save_external_model_asset(
        model_identifier=experiment_name,
        name=experiment_name,
        schemas=external_schemas,
        training_data_reference=trainingdataref,
        catalog_id=catalog_id,
        description="Customer Churn Model with XGBOOST",
    )

    muc_utilities = facts_client.assets.get_ai_usecase(
        ai_usecase_id=model_entry_id,
        catalog_id=catalog_id,
    )

    fs_model.track(
        usecase=muc_utilities,
        approach=muc_utilities.get_approaches()[0],
        version_number="minor",  # "0.1.0"
    )
