# SPDX-License-Identifier: Apache-2.0
import os
import joblib
import re

from ibm_watson_machine_learning import APIClient
from mlmonitor.src.wml import wml_client as WML_CLIENT, WML_SPACE_ID
from mlmonitor.src import API_KEY, logger, DATA_ROOT
from mlmonitor.src.utils.validation import validate_hdf_file, is_csv
from mlmonitor.src.utils.file_utils import make_model_tgzfile
from mlmonitor.src.model.config_wml import WMLModelconfig
from mlmonitor.data import BUCKET_NAME, COS_ENDPOINT, COS_RESOURCE_CRN


def create_model_asset(
    fs_client,
    model_data: str,
    deployment_name: str,
    model_config: WMLModelconfig,
    ibm_api_key: str = API_KEY,
    wml_client: APIClient = WML_CLIENT,
    wml_space_id: str = WML_SPACE_ID,
) -> dict:
    """
    create a model asset within the deployment space

    Args:
        fs_client: instantiated Factsheets client to prepare_model_meta
        model_data: local model file path
        deployment_name: deployment name
        model_config: model configuration
        ibm_api_key: Cloud Pak for Data API Key Defaults to API_KEY.
        wml_client: Watson Machine Learning API Defaults to WML_CLIENT.
        wml_space_id: CP4D Deployment space Defaults to WML_SPACE_ID.

    Raises:
        ValueError: data type error

    Returns:
        Published model UID
    """
    ##################################################
    # PREPARE MODEL UPLOAD DEPENDING ON ML FRAMEWORK #
    ##################################################
    framework = model_config.inference_framework

    if framework == "scikit-learn":
        logger.info(f"{framework} => Reloading model from joblib")
        model = joblib.load(model_data)

    elif framework == "tensorflow":
        logger.info(f"{framework} => Generate model archive for upload")
        is_h5 = validate_hdf_file(model_data)

        if not is_h5:
            raise ValueError("Invalid model file should be a valid hdf (.h5) file")

        model_details = re.match(r"^(/.*)/(.*).h5$", model_data)
        output_tgz_name = f"{model_details[2]}.tgz"
        model_filename = f"{model_details[2]}.h5"
        model_path = model_details[1]
        model = make_model_tgzfile(
            output_filename=output_tgz_name,
            source_dir=model_path,
            filename=model_filename,
        )

    elif framework == "xgboost":
        logger.info(f"{framework} => Reloading model from joblib")
        model = joblib.load(model_data)
    else:
        raise ValueError('Invalid framework ["scikit-learn", "tensorflow", "xgboost"]')

    ######################################################################
    # RETRIEVE MODEL USE CASE SPECIFIC CONFIGURATION DETAILS (SIGNATURE) #
    ######################################################################
    data_type = model_config.data_type

    framework_version = model_config.inference_framework_version
    target = model_config.class_label

    train_data_path = model_config._get_data_location(dataset_type="train")
    dataset_is_csv = is_csv(train_data_path)

    dataset = model_config._get_data(dataset_type="train")

    if data_type == "structured":

        scoring_df = dataset.loc[:, model_config.feature_columns]
        labels = dataset[target]

    elif data_type == "unstructured_image":

        samples, labels = dataset
        logger.info("samples shape sent for inference", samples.shape)

        scoring_df = None
        labels = None

    wml_client.set.default_space(space_uid=wml_space_id)

    ##########################
    # Create Model Asset     #
    ##########################

    software_spec_uid = wml_client.software_specifications.get_id_by_name(
        model_config.inference_instance
    )
    logger.info(
        f"Software Specification {model_config.inference_instance} ID: {software_spec_uid}"
    )

    model_props = {
        wml_client._models.ConfigurationMetaNames.NAME: deployment_name,
        wml_client._models.ConfigurationMetaNames.TYPE: f"{framework}_{framework_version}",
        wml_client._models.ConfigurationMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
    }

    ###############################################################################
    # Create TRAINING DATA REFERENCE  (COS Bucket only) for structured datasets   #
    ###############################################################################
    if data_type == "structured" and dataset_is_csv:
        from mlmonitor.data.cos import put_item
        from mlmonitor.src import IAM_URL
        from mlmonitor.data import CLOUD_API_KEY

        # TODO training_data_references for CP4D on prem should be DB2
        put_item(
            item_name=model_config.training_data,
            item_path=os.path.join(DATA_ROOT, model_config.data_dir),
        )
        datasource_type = wml_client.connections.get_datasource_type_uid_by_name(
            "bluemixcloudobjectstorage"
        )
        conn_meta_props = {
            wml_client.connections.ConfigurationMetaNames.NAME: f"COS Connection for {deployment_name}",
            wml_client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: datasource_type,
            wml_client.connections.ConfigurationMetaNames.DESCRIPTION: "Connection to COS",
            wml_client.connections.ConfigurationMetaNames.PROPERTIES: {
                "bucket": BUCKET_NAME,
                "api_key": CLOUD_API_KEY,
                "resource_instance_id": COS_RESOURCE_CRN,
                "iam_url": IAM_URL,
                "url": COS_ENDPOINT,
            },
        }

        # Data source details
        conn_details = wml_client.connections.create(meta_props=conn_meta_props)
        connection_id = wml_client.connections.get_uid(conn_details)

        training_data_references = [
            {
                "id": deployment_name,
                "type": "connection_asset",
                "connection": {
                    "id": connection_id,
                    "href": f"/v2/connections/{connection_id}?space_id={wml_space_id}",
                },
                "location": {
                    "bucket": BUCKET_NAME,
                    "file_name": model_config.training_data,
                },
            }
        ]

        model_props[
            wml_client._models.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
        ] = training_data_references
        model_props[wml_client._models.ConfigurationMetaNames.LABEL_FIELD] = target

    ##########################
    # Create Model Asset     #
    ##########################
    fs_client.export_facts.prepare_model_meta(wml_client, model_props)

    logger.info(f"Storing model to deployment space {wml_space_id}")
    published_model_details = wml_client.repository.store_model(
        model=model,
        meta_props=model_props,
        training_data=scoring_df,
        training_target=labels,
    )

    model_uid = wml_client.repository.get_model_id(published_model_details)
    logger.info(f"Model Asset creation Completed with Model ID: {model_uid}")

    if data_type == "unstructured_image":
        os.remove(model)

    return model_uid


def deploy_wml_endpoint(
    deployment_name: str,
    model_config: WMLModelconfig,
    model_uid: str,
    wml_client: APIClient = WML_CLIENT,
) -> dict:
    """
    Triggers a WML Endpoint Deployment for a given model use case
    and register model deployment details AI Factsheets

    :param deployment_name:str: the Endpoint Name of the new deployed model
    :param model_config:WMLModelConfig: WML model config Object
    :param model_uid: WML model asset ID
    :param wml_client:APIClient: instantiated Watson Machine Learning Client
    :return: A dictionary with the model endpoint name and source directory
    """
    num_samples = 2
    data_type = model_config.data_type

    target = model_config.class_label
    features = model_config.feature_columns
    dataset = model_config._get_data(num_samples=num_samples, dataset_type="train")

    if data_type == "structured":

        scoring_df = dataset.loc[:, model_config.feature_columns]

        scoring_payload = {
            "input_data": [
                {
                    "fields": scoring_df.columns.to_list(),
                    "values": scoring_df.head(num_samples).values.tolist(),
                }
            ]
        }

    elif data_type == "unstructured_image":

        samples, labels = dataset
        print("samples shape sent for inference", samples.shape)

        scoring_payload = {"input_data": [{"values": samples.tolist()}]}

    ##########################
    # Create WML Deployment  #
    ##########################

    deployment_details = wml_client.deployments.create(
        model_uid,
        meta_props={
            wml_client.deployments.ConfigurationMetaNames.TAGS: ["mlmonitor"],
            wml_client.deployments.ConfigurationMetaNames.NAME: deployment_name,
            wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
        },
    )

    deployment_uid = wml_client.deployments.get_uid(deployment_details)
    scoring_url = wml_client.deployments.get_scoring_href(deployment_details)
    logger.info(
        f"Scoring URL:{scoring_url}\nModel id: {model_uid}\nDeployment id: {deployment_uid}"
    )

    ##########################
    # Sample Scoring Part    #
    ##########################

    logger.debug(f"Scoring request WML: {scoring_payload}")
    scoring_response = wml_client.deployments.score(deployment_uid, scoring_payload)
    logger.info(
        f"Raw Predictions received for {num_samples} samples:\n{scoring_response}"
    )

    return {
        "model_endpoint": deployment_name,
        "features": features,
        "target": target,
        "source_dir": model_config._source_dir,
    }
