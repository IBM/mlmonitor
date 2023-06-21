# SPDX-License-Identifier: Apache-2.0
import os
import boto3
import json
from sagemaker import serializers, deserializers
from typing import Optional


from mlmonitor.src import (
    MODEL_ENTRY_ID,
    CATALOG_ID,
    logger,
    get_aws_credentials,
    API_KEY,
    USERNAME,
    AUTH_ENDPOINT,
    VERIFY_CP4D_SSL,
)
from mlmonitor.src.aws import (
    sagemaker_models,
    sagemaker_serializers,
    sagemaker_deserializers,
)
from mlmonitor.src.model.config_aws import SageMakerModelConfig
from mlmonitor.src.aws.deployment import generate_base_deployment_params
from mlmonitor.src.factsheets.deployments import add_aws_deployment_details
from mlmonitor.src.factsheets.utils import (
    get_model_id_by_model_name,
    get_model_id_by_deployment_name,
    FactsheetHelpers,
)


def deploy_sagemaker_endpoint(
    model_data: str,
    deployment_name: str,
    model_config: SageMakerModelConfig,
    ibm_key_name: str,
) -> dict:
    """
    Triggers a Sagemaker Endpoint Deployment for a given model use case
    Make sure to set AWS credentials in environment variables :
    `AWS_ACCESS_KEY_ID`: aws key
    `AWS_SECRET_ACCESS_KEY`:  credentials
    `AWS_DEFAULT_REGION`:  aws region

    :param model_data: this indicates the s3 url path of model data to be deployed, if produced by a training job ,
     format looks like: s3://sagemaker-<region>-1-<account>/sagemaker/<prefix>/<job name>/output/model.tar.gz
    :param deployment_name: this boolean flag indicates whether a custom training script and source directory should be used to train the model
    :param model_config: AWS model config Object
    :param ibm_key_name: Name of IBM api key to be used by Sagemaker (needed for Payload logging to Watson OpenScale)
    :param output_model_data: directory output Path where dictionary with model details in model_data.json file should be written
    :return: dictionary with model use case details
      {
        "features": [],
        "job_name": "<sagemaker training job name>",
        "model_data": "<model.tar.gz location>",
        "model_endpoint": "<sagemaker endpoint name if already>",
        "target": "<model target column>"
        }
    """

    ######################################################################
    # RETRIEVE MODEL USE CASE SPECIFIC CONFIGURATION DETAILS (SIGNATURE) #
    ######################################################################

    inference_samples = 2
    framework = model_config.inference_framework
    target = model_config.class_label
    features = model_config.feature_columns

    dataset = model_config._get_data(
        num_samples=inference_samples, dataset_type="validation"
    )

    model_params = generate_base_deployment_params(
        trained_model_data=model_data,
        framework=framework,
        source_dir=model_config.source_dir,
        framework_version=model_config.inference_framework_version,
        py_version=model_config.inference_py_version,
        script=model_config.inference_script,
        instance=model_config.inference_instance,
    )

    ######################################################################
    # CONFIGURE SAGEMAKER INFERENCE RUNTIME BASED ON MODEL SIGNATURE     #
    ######################################################################

    SelectedModel = sagemaker_models.get(framework)
    SelectedDeserializer = sagemaker_deserializers.get(
        model_config.deserializer, deserializers.JSONDeserializer
    )
    SelectedSerializer = sagemaker_serializers.get(
        model_config.serializer, serializers.JSONSerializer
    )

    logger.info(
        f"endpoint name :[{deployment_name}]\n"
        f"ibm key name :[{ibm_key_name}]\n"
        f"model data  :[{model_data}]\n"
        f"container  :[{model_params.get('image_uri')}]\n"
        f"region  :[{os.environ.get('AWS_DEFAULT_REGION')}]\n"
    )

    logger.info(f"model_params:\n{json.dumps(model_params, indent=4)}")
    logger.info(f"Deploying inference Endpoint {deployment_name} to AWS")

    selected_model = SelectedModel(**model_params)

    predictor = selected_model.deploy(
        endpoint_name=deployment_name,
        initial_instance_count=1,
        instance_type=model_config.inference_instance,
        serializer=SelectedSerializer(),
        deserializer=SelectedDeserializer(),
    )

    deployment_name = predictor.endpoint

    ############################
    # SCORE AWS ENDPOINT       #
    ############################

    if model_config.data_type == "structured":
        inputs = dataset.loc[:, model_config.feature_columns].to_numpy()
        scoring_data = {"instances": [{"features": x} for x in inputs.tolist()]}
    else:
        values, labels = dataset
        scoring_data = {"input_data": [{"values": values.tolist()}]}
        logger.info(f"data shape sent to model : {values.shape}")

    preds = predictor.predict(scoring_data)
    logger.info(f"Raw Predictions received for {inference_samples} samples:\n{preds}")

    return {
        "model_endpoint": deployment_name,
        "model_data": model_data,
        "features": features,
        "deployment_target": "aws",
        "target": target,
        "source_dir": model_config.source_dir,
    }


def govern_sagemaker_endpoint(
    deployment_name: str,
    model_config: SageMakerModelConfig,
    model_entry_id: str = MODEL_ENTRY_ID,
    catalog_id: str = CATALOG_ID,
    env: str = "saas",
    cp4d_username: Optional[str] = None,
    cp4d_url: Optional[str] = None,
):
    """
    Adds an AI Factsheets deployment details for AWS model endpoint entry for this deployment.
    Links (relatemodels API call) the newly created asset with the existing model asset generated at training time by Sagemaker training job.
    :param deployment_name:str: deployment name corresponding to AWS Endpoint
    :param model_config: AWS model config Object
    :param model_entry_id:str=MODEL_ENTRY_ID: AI Factsheets mode entry ID
    :param catalog_id:str=CATALOG_ID: AI Factsheets mode catalog ID
    :param env:str='saas' Cloud Pak for Data Username for on prem environment
    :param cp4d_username:Optional[str] Cloud Pak for Data Username for on prem environment
    :param cp4d_url:Optional[str]: Cloud Pak for Data URL for on prem environment
    :return:
    """

    key, secret, region, role = get_aws_credentials()
    session = boto3.session.Session(
        aws_access_key_id=key, aws_secret_access_key=secret, region_name=region
    )

    ######################################################################
    # RETRIEVE MODEL USE CASE SPECIFIC CONFIGURATION DETAILS (SIGNATURE) #
    ######################################################################

    assert (framework := model_config.inference_framework) in [
        "pytorch",
        "sklearn",
        "tensorflow",
        "xgboost",
    ]

    SelectedModel = sagemaker_models.get(framework)

    logger.info(f"Model use case id [{model_entry_id}]\ncatalog id [{catalog_id}]\n")

    add_aws_deployment_details(
        apikey=API_KEY,
        session=session,
        endpoint_name=deployment_name,
        model_entry_id=model_entry_id,
        catalog_id=catalog_id,
        inference_entrypoint=model_config.inference_script,
        source_dir=model_config.source_dir,
        description=model_config.description,
        framework=SelectedModel.__name__,
        framework_version=model_config.inference_framework_version,
        env=env,
        cp4d_username=cp4d_username,
        cp4d_url=cp4d_url,
    )

    if model_entry_id and catalog_id:
        # TODO relate models with AIGOV client
        # https://cloud.ibm.com/apidocs/factsheets#updatemasterid
        fs_helper = FactsheetHelpers(
            api_key=API_KEY,
            container_type="catalog",
            container_id=catalog_id,
            model_entry_id=model_entry_id,
            username=USERNAME,
            cpd_url=AUTH_ENDPOINT,
            env=env,
        )

        models = fs_helper.get_models(verify=VERIFY_CP4D_SSL)
        # get model id in development state
        reference_model_id = get_model_id_by_model_name(
            models=models, model_name=deployment_name, state="development"
        )

        # get model id with a deployment
        model_id = get_model_id_by_deployment_name(
            models=models, deployment_name=deployment_name
        )

        if reference_model_id and model_id:
            logger.info(
                f"Model with name {deployment_name} exists in Factsheets with model id: [{reference_model_id}]\n"
                f"Deployment with name {deployment_name} exists in Factsheets under model id [{model_id}]\n"
                f"Relate the 2 models in Factsheets Reference[{reference_model_id}]=> Target[{model_id}]"
            )

            resp = fs_helper.relatemodels(
                reference_model_id=reference_model_id,
                model_id=model_id,
                verify=VERIFY_CP4D_SSL,
            )
            logger.info(resp)
