# SPDX-License-Identifier: Apache-2.0
import joblib
import os
import json
import time
import logging

import pandas as pd
import sys
import numpy
import argparse
from typing import Optional
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, get_scorer
from sklearn.model_selection import cross_validate
from ibm_aigov_facts_client import (
    AIGovFactsClient,
    ModelEntryProps,
    ExternalModelSchemas,
    TrainingDataReference,
    CloudPakforDataConfig,
)

numpy.set_printoptions(threshold=sys.maxsize)


def init_logger(level: int = logging.INFO) -> logging.Logger:
    """
    initializes the logger object for this module.
    It sets the log level to DEBUG and adds a StreamHandler to write output
    to sys.stdout

    :return: initialized logger instance
    """
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s gcr module] : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


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
    region_name = params.get("region_name")
    API_KEY_NAME = params.get("ibm_key_name")
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
        PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        try:
            from use_case_churn.utils import git_branch

            fs_tags = {"git_branch": git_branch(logger=logger, path=PROJECT_ROOT)}
        except ModuleNotFoundError as e:
            print(f"Git branch error {e}")
            fs_tags = {}

        time_suffix = time.strftime("%Y%m%d-%H%M", time.gmtime())
        experiment_name = os.getenv("JOB_NAME", f"gcr-scikit-gbc-{time_suffix}")
    # Init FactSheets Client
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
    print(
        f'initialize ibm_aigov_facts_client version : [{facts_client.version}] - env: [{params.get("cp4d_env")}]'
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
    logger.info(f"Current RunID {run_id}")

    if len(metrics) > 0:
        logger.info("facts_client.runs.log_metrics FAILING - skip")
        # facts_client.runs.log_metrics(run_id, metrics)
        logger.debug(f"factsheets metrics:\n{json.dumps(metrics, indent=4)}")
    if len(tags) > 0:
        facts_client.runs.set_tags(run_id, tags)
        logger.debug(f"factsheets tags:\n{json.dumps(tags, indent=4)}")
    if len(params) > 0:
        facts_client.runs.log_params(run_id, params)
        logger.debug(f"factsheets params:\n{json.dumps(params, indent=4)}")

    facts_client.export_facts.export_payload(run_id)

    if inputs and outputs:
        external_schemas = ExternalModelSchemas(input=inputs, output=outputs)

    trainingdataref = TrainingDataReference(schema=tdataref) if tdataref else None

    fs_model = facts_client.external_model_facts.save_external_model_asset(
        model_identifier=experiment_name,
        catalog_id=catalog_id,
        name=experiment_name,
        schemas=external_schemas,
        training_data_reference=trainingdataref,
        description="Scikit Credit Risk model",
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


def fetch_dataset(data_path: str, filename: str = "training.csv") -> pd.DataFrame:
    # Take the set of files and read them all into a single pandas dataframe
    print(f"fetch_dataset {os.listdir(data_path)}")
    input_files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if filename in file
    ]

    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(data_path, "train")
        )
    raw_data = [pd.read_csv(file, engine="python") for file in input_files]

    return pd.concat(raw_data)


def train_wml(
    model_dir: str,
    data_path: str,
    train_dataset: str,
    val_dataset: Optional[str] = None,
    test_dataset: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **hyperparameters,
) -> str:
    """
    train_wml is used to train this model in local environment
    this python module `use_case_gcr` is imported dynamically by `mlmonitor`

    - this function definition should always contain as input parameters :
    model_dir , data_path , train_dataset,val_dataset,test_dataset ,logger ,and  hyperparameters as parameters

    - this function must produce a model artifact return its location in model_data pah

    .. code-block:: python
        :caption: Example
        from mlmonitor import WMLModelUseCase
        model_use_case = WMLModelUseCase(source_dir='use_case_gcr', catalog_id=catalog_id, model_entry_id=model_entry_id)
        model_use_case.train() => this function is invoked by trained task

    :param model_dir:str: Base directory where to store model after training completion
    :param data_path:str: location (directory path) of the datasets for this model use case
    :param train_dataset:str: filename of training dataset
    :param val_dataset:Optional[str]=None:  filename of validation dataset
    :param test_dataset:Optional[str]=None:  filename of test dataset
    :param logger:Optional[logging.Logger]=None: Pass instantiated logger object
    :param **hyperparameters: model hyperparameters to use for model training task
    :return: path to the model artifact produced
    """
    train_data = fetch_dataset(data_path=data_path)
    model_data = train(
        model_dir=os.path.join(model_dir, "model_gcr"),
        train_data=train_data,
        logger=logger,
    )

    return model_data


def train(
    model_dir: str, train_data: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> str:
    """
    scikit-learn model trainer.

    :param model_dir:str: Specify the directory where the model should be saved
    :param train_data:pd.DataFrame: Pass in the training data as Pandas DatFrame
    :param logger:Optional[logging.Logger]=None: logger passed to the function
    :return: model path
    """
    target = "Risk"
    feature_cols = train_data.drop(columns=[target])
    labels = train_data[target]

    # Set model evaluation properties
    optimization_metric = "roc_auc"
    random_state = 33
    holdout_fraction = 0.1

    if type_of_target(labels.values) in ["multiclass", "binary"]:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            feature_cols,
            labels,
            test_size=holdout_fraction,
            random_state=random_state,
            stratify=labels.values,
        )
    else:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            feature_cols, labels, test_size=holdout_fraction, random_state=random_state
        )

    # Data preprocessing transformer generation
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("OrdinalEncoder", OrdinalEncoder(categories="auto", dtype=numpy.float64)),
        ]
    )

    numeric_features = feature_cols.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = feature_cols.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Initiate model and create pipeline
    model1 = GradientBoostingClassifier()
    gbt_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", model1)]
    )
    model_gbt = gbt_pipeline.fit(X_train, y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model_gbt, model_path)
    # np.save(file=os.path.join(params.get('model_dir'), "feature_cols.npy"), arr=feature_cols)

    y_pred = model_gbt.predict(X_holdout)

    # Evaluate model performance on test data and Cross validation
    scorer = get_scorer(optimization_metric)
    scorer(model_gbt, X_holdout, y_holdout)

    # Cross validation -3 folds
    cv_results = cross_validate(
        model_gbt,
        X_train,
        y_train,
        cv=3,
        scoring={optimization_metric: scorer},
    )
    numpy.mean(cv_results[f"test_{optimization_metric}"])

    logger.info(f"\n{classification_report(y_pred, y_holdout)}")

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    # CP4D specific arguments
    parser.add_argument("--catalog-id", type=str)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--model-entry-id", type=str)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--ibm-key-name", type=str,
                        default="IBM_API_KEY_MLOPS")  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-env", type=str, default=os.getenv("ENV", "saas"),
                        choices=["saas", "prem"], )  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-username", type=str, default=None)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--cp4d-url", type=str, default=None)  # used by train_sagemaker_job,train_az_ml_job
    parser.add_argument("--model-name", type=str, default="gcr-model")

    # Training Job specific arguments (Sagemaker,Azure,WML) default SageMaker envar or Azure expected values
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./outputs"))
    parser.add_argument("--output-data-dir", type=str, default=os.getenv("SM_OUTPUT_DATA_DIR", "./outputs"))

    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TEST"))
    parser.add_argument("--validation", type=str, default=os.getenv("SM_CHANNEL_VALIDATION"))

    parser.add_argument("--region-name", type=str, default="ca-central-1")
    # fmt: on

    args = parser.parse_args()
    parameters = vars(args)
    log_level = int(os.getenv("LOG_LEVEL", logging.INFO))
    logger = init_logger(level=log_level)
    (
        facts_client,
        props,
        EXPERIMENT_NAME,
        EXPERIMENT_ID,
        tags,
        params,
    ) = init_external_fs_client(logger=logger, **parameters)

    print(f'training {os.environ.get("SM_CHANNEL_TRAINING")}')
    print(f'train {os.environ.get("SM_CHANNEL_TRAIN")}')
    print(f"params {params}")
    print(f"parameters {parameters}")
    print(f'SM_CHANNEL_TRAIN {os.environ.get("SM_CHANNEL_TRAIN")}')
    print(f'SM_CHANNEL_TRAINING {os.environ.get("SM_CHANNEL_TRAINING")}')
    train_data = fetch_dataset(data_path=parameters.get("train"))

    train(model_dir=parameters.get("model_dir"), train_data=train_data, logger=logger)
    save_fs_model(
        logger=logger,
        catalog_id=parameters.get("catalog_id"),
        model_entry_id=parameters.get("model_entry_id"),
        facts_client=facts_client,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        tags=tags,
        params=params,
    )
