# SPDX-License-Identifier: Apache-2.0
import boto3
import sagemaker
import os
from typing import Optional
from sagemaker.serializers import CSVSerializer
import pandas as pd

from mlmonitor.src.aws import sagemaker_estimators
from mlmonitor.src.aws.utils import (
    jobname_from_modeldata,
    s3_upload_training_job_datasets,
)
from mlmonitor.src.aws.training import (
    generate_xgb_training_params,
    generate_sk_training_params,
    generate_pt_training_params,
    generate_tf_training_params,
)
from mlmonitor.src import (
    ROLE,
    DATA_ROOT,
    PROJECT_ROOT,
    MODEL_ENTRY_ID,
    CATALOG_ID,
    logger,
    get_aws_credentials,
)
from mlmonitor.src.model.config_aws import SageMakerModelConfig


def train_sagemaker_job(
    model_config: SageMakerModelConfig,
    custom: bool,
    ibm_key_name: str = "IBM_API_KEY_MLOPS",
    model_entry_id: str = MODEL_ENTRY_ID,
    catalog_id: str = CATALOG_ID,
    deploy: str = False,
    data_path: str = DATA_ROOT,
    cp4d_env: str = "saas",
    cp4d_username: Optional[str] = None,
    cp4d_url: Optional[str] = None,
) -> dict:
    """
    Triggers a Sagemaker training job for a given model use case and register Training metadata to AI Factsheets service in specified catalog id / Model use case id

    Make sure to set AWS credentials in environment variables :
    `AWS_ACCESS_KEY_ID`: aws key
    `AWS_SECRET_ACCESS_KEY`:  credentials
    `AWS_DEFAULT_REGION`:  aws region

    :param model_config: AWS model config Object
    :param custom: this boolean flag indicates whether a custom training script and source directory should be used to train the model
    :param ibm_key_name: Name of IBM api key to be used by Sagemaker (needed for Payload logging to Watson OpenScale)
    :param model_entry_id: IBM AI Factsheets Model use case identifier to be used to register the deployed model
    :param catalog_id: IBM AI Factsheets catalog identifier to be used to register the deployed model
    :param deploy: This boolean flag indicates whether produced model archive should be deployed to a Sagemaker endpoint
    :param data_path: directory Path where train and validation dataset should be located. default is Project datasets folder
    :param cp4d_env:str=saas Cloud Pak for Data environment where Factsheets service is located
    :param cp4d_username:str=None Cloud Pak for Data username for on premise environment
    :param cp4d_url:str=None Cloud Pak for Data url for on premise environment
    :return: dictionary with model use case details
        {
        "features": [],
        "job_name": "<sagemaker training job name>",
        "model_data": "<model.tar.gz location produce in s3>",
        "target": "<model target column>"
        "deployment_target": "aws"
        }
    """
    key, secret, region, role = get_aws_credentials()
    session = boto3.session.Session(
        aws_access_key_id=key, aws_secret_access_key=secret, region_name=region
    )
    region = session.region_name
    sagemaker_session = sagemaker.Session(session)
    bucket = sagemaker_session.default_bucket()

    train_dict = s3_upload_training_job_datasets(
        data_path=data_path,
        sagemaker_session=sagemaker_session,
        boto_session=session,
        bucket=bucket,
        prefix=model_config.prefix,
        training_data_location=model_config.training_data,
        val_data_location=model_config.validation_data,
        test_data_location=model_config.test_data,
    )

    entry_point = model_config.train_script if custom else None
    source_dir = os.path.join(PROJECT_ROOT, model_config.source_dir) if custom else None

    # Mandatory script parameters for each model use case
    system_params = {
        "ibm-key-name": ibm_key_name,
        "catalog-id": catalog_id,
        "model-entry-id": model_entry_id,
        "cp4d-env": cp4d_env,
    }
    if cp4d_env == "prem":  # cp4d-url and cp4d-username only needed for CP4D on-prem
        system_params["cp4d-url"] = cp4d_url
        system_params["cp4d-username"] = cp4d_username
    # Specific hyperparameters passed in model signature config file
    hparam = (
        {**system_params, **model_config.hyperparameters}
        if custom
        else model_config.hyperparameters
    )

    base_estimator_params = {
        "base_job_name": model_config.base_job_name,
        "entry_point": entry_point,
        "source_dir": source_dir,
        "role": ROLE,
        "instance_count": 1,
        "instance_type": model_config.train_instance,
        "output_path": f"s3://{bucket}/{model_config.prefix}",
        "hyperparameters": hparam,
    }

    # framework specifics updates

    if (framework := model_config.train_framework) == "xgboost":
        estimator_params, train_dict = generate_xgb_training_params(
            framework=framework,
            framework_version=model_config.train_framework_version,
            estimator_params=base_estimator_params,
            train_dict=train_dict,
            sagemaker_session=sagemaker_session,
        )
    elif framework == "sklearn":
        estimator_params, train_dict = generate_sk_training_params(
            framework=framework,
            framework_version=model_config.train_framework_version,
            estimator_params=base_estimator_params,
            train_dict=train_dict,
            sagemaker_session=sagemaker_session,
        )
    elif framework == "tensorflow":
        estimator_params, train_dict = generate_tf_training_params(
            py_version=model_config.train_py_version,
            framework_version=model_config.train_framework_version,
            estimator_params=base_estimator_params,
            train_dict=train_dict,
        )
    elif framework == "pytorch":
        estimator_params, train_dict = generate_pt_training_params(
            py_version=model_config.train_py_version,
            framework_version=model_config.train_framework_version,
            estimator_params=base_estimator_params,
            train_dict=train_dict,
        )
    else:
        raise ValueError(
            f"expecting model-type in xgboost,sklearn,tensorflow or pytorch but {framework} passed"
        )

    SelectedEstimator = sagemaker_estimators.get(framework)

    logger.info(
        f"Train model\n"
        f"with role : [{ROLE}]\n"
        f"custom train : [{custom}]\n"
        f"job name : [{model_config.base_job_name}]\n"
        f"deploy model : [{deploy}]\n"
        f"train entrypoint : [{entry_point}]\n"
        f"source_dir : [{source_dir}]\n"
        f"Model use case id [{model_entry_id}]\n"
        f"catalog id [{catalog_id}]\n"
        f"ibm key name [{ibm_key_name}]\n"
        f"AWS Region [{region}]\n"
        f"SM version [{sagemaker.__version__}]\n"
        f"Estimator class [{SelectedEstimator.__name__}]\n"
        f"Estimator parameters \n[{estimator_params}]\n"
        f"Script parameters \n[{hparam}]\n"
    )

    est = SelectedEstimator(**estimator_params)
    est.fit(train_dict)

    trained_model_data = est.model_data
    job_name = jobname_from_modeldata(model_data=trained_model_data)
    # job_name = est.training_job_analytics.name

    logger.info(f"Model artifact produced by training job {trained_model_data}")

    deployment_name = jobname_from_modeldata(model_data=trained_model_data)

    model_data = {
        "model_data": trained_model_data,
        "job_name": job_name,
        "features": model_config.feature_columns,
        "target": model_config.class_label,
    }

    # optional deployment section
    if deploy:
        predictor = simple_deploy(
            features=tuple(model_config.feature_columns),
            dataset=model_config.validation_data,
            estimator=est,
            deployment_name=deployment_name,
        )

        model_data["model_endpoint"] = predictor.endpoint
        model_data["deployment_target"] = "aws"

    return model_data


def simple_deploy(
    features: tuple, dataset: str, estimator: sagemaker.estimator, deployment_name: str
):
    test_data = pd.read_csv(dataset, engine="python")
    inference_samples = 2

    inputs = test_data.loc[0:inference_samples, features].to_numpy()

    predictor = estimator.deploy(
        initial_instance_count=1,
        endpoint_name=deployment_name,
        instance_type="ml.m4.xlarge",
        serializer=CSVSerializer(),
    )

    preds = predictor.predict(inputs).decode("utf-8").split("\n")[:-1]
    logger.info(f"predictions {preds}")

    return predictor
