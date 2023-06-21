# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale.supporting_classes.enums import DataSetTypes, TargetTypes
import itertools
from typing import Dict

from mlmonitor.src.wos import wos_client
from mlmonitor.src import logger
from mlmonitor.src.model.config import ModelConfig
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment


def log_feedback_data(
    model_config: ModelConfig,
    deployment_name: str,
    deployment_target: str = "aws",
    inference_samples: int = 2,
    include_predictions: bool = False,
    dataset_type: str = "validation",
) -> Dict:
    """
    send <inference_samples> Feedback logging requests to the deployed Sagemaker,Azure or WML Endpoint with name <deployment_name>

    :param model_config: ModelConfig: model configuration
    :param deployment_name: str: Endpoint Name in Sagemaker Azure or WML for which an OpenScale subscription should be created
    :param deployment_target: str: target model serving environment wml , aws , azure
    :param inference_samples: int: Number of inference samples to send in the scoring request to the deployed Endpoint (deployment_name)
    :param include_predictions: bool: whether include predictions in feedback logging or let WOS score endpoint at evaluation
    :param dataset_type: str: dataset type to use for payload logging test , train , validation
    :return: Dict dictionary with payload_dataset_id,payload_sent,payload_processed,success keys
    """
    assert deployment_target in {
        "wml",
        "aws",
        "azure",
    }, "Deployment target should be wml aws or azure"
    assert dataset_type in {"test", "train", "validation"}

    ###########################
    # VERIFY WOS SUBSCRIPTION #
    ###########################
    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=deployment_name
    )

    if len(subscription_ids) != 1:
        raise ValueError(
            f"{deployment_name} has  {len(subscription_ids)} subscription(s) , 1 expected"
        )

    subscription_id = subscription_ids[0]

    logger.debug(
        f"Deployment Name {deployment_name}\n" f"subscription_id {subscription_id}\n"
    )

    #####################################
    # PREPARE FEEDBACK LOGGING REQUESTS #
    #####################################
    dataset = model_config._get_data(
        num_samples=inference_samples, dataset_type=dataset_type
    )
    logger.info(
        f"Feedback Logging for {deployment_name} in {deployment_target} on {dataset_type} dataset"
    )

    if deployment_target == "aws":
        from mlmonitor.src import aws_credentials
        from mlmonitor.src.aws.scoring import get_scores_labels, _score_unstructured
    elif deployment_target == "wml":
        from mlmonitor.src.wml import wml_client, WML_SPACE_ID

        wml_client.set.default_space(WML_SPACE_ID)
        from mlmonitor.src.wml.scoring import get_scores_labels, _score_unstructured
    elif deployment_target == "azure":
        from mlmonitor.src.azure import AZ_WORKSPACE
        from mlmonitor.src.azure.scoring import get_scores_labels, _score_unstructured

    else:
        raise NotImplementedError("deployment_target should be aws or wml")

    if model_config.data_type == "structured":

        request_df = dataset
        inference_samples = min(inference_samples, request_df.shape[0])

        if include_predictions:
            scoring_df = request_df.loc[:, model_config.feature_columns]

            # Feedback request with predictions
            wos_get_scores_labels_params = {
                "df": scoring_df,
                "endpoint_name": deployment_name,
            }

            if deployment_target == "aws":
                wos_get_scores_labels_params = {
                    **wos_get_scores_labels_params,
                    **aws_credentials,
                }
            elif deployment_target == "wml":
                # TODO implement get_scores_labels for wml
                wos_get_scores_labels_params["wml_client"] = wml_client
            elif deployment_target == "azure":
                wos_get_scores_labels_params = {
                    **wos_get_scores_labels_params,
                    "workspace": AZ_WORKSPACE,
                }

            else:
                raise NotImplementedError("deployment_target should be aws or wml")

            proba_vectors, proba_scores, predicted_labels = get_scores_labels(
                **wos_get_scores_labels_params
            )

            scoring_feedback_wos = {
                "fields": request_df.columns.tolist()
                + ["_original_prediction", "_original_probability"],
                "values": [
                    x + [y] + [z]
                    for x, y, z in zip(
                        request_df.values.tolist(),
                        predicted_labels.tolist(),
                        proba_vectors.tolist(),
                    )
                ],
            }
        else:
            # Feedback request without predictions
            scoring_feedback_wos = {
                "fields": request_df.columns.tolist(),
                "values": request_df.values.tolist(),
            }

    elif model_config.data_type == "unstructured_image":

        samples, labels = dataset

        scoring_input_target = [
            [x, y] for x, y in zip(samples.tolist(), labels.tolist())
        ]

        if include_predictions:

            if deployment_target == "aws":
                scoring_response_wos = _score_unstructured(
                    payload=samples,
                    endpoint_name=deployment_name,
                    **aws_credentials,
                )
            elif deployment_target == "azure":
                scoring_response_wos = _score_unstructured(
                    payload=samples,
                    endpoint_name=deployment_name,
                    workspace=AZ_WORKSPACE,
                )
            elif deployment_target == "wml":
                scoring_response_wos = _score_unstructured(
                    payload=samples, endpoint_name=deployment_name, client=wml_client
                )
            else:
                raise NotImplementedError("deployment_target should be aws or wml")

            predictions = scoring_response_wos.get("predictions")[0].get("values")
            print("Logging Feedback data WITH predictions included")
            values = [
                list(itertools.chain(*i))
                for i in zip(scoring_input_target, predictions)
            ]
            fields = [
                "scoring_input",
                "target",
                "_original_prediction",
                "_original_probability",
            ]

        else:
            # Feedback request without predictions
            values = scoring_input_target
            fields = ["scoring_input", "target"]

        scoring_feedback_wos = {"fields": fields, "values": values}

    else:
        raise ValueError("data_type should be structured (csv) or unstructured_image ")

    logger.debug(f"openscale feedback request : {scoring_feedback_wos}")

    # Datasets FEEDBACK_DATASET
    feedback_data_set_id = (
        wos_client.data_sets.list(
            type=DataSetTypes.FEEDBACK,
            target_target_id=subscription_id,
            target_target_type=TargetTypes.SUBSCRIPTION,
        )
        .result.data_sets[0]
        .metadata.id
    )
    assert feedback_data_set_id
    logger.info(f"Feedback data set id {feedback_data_set_id}")

    fl_records_before = wos_client.data_sets.get_records_count(feedback_data_set_id)

    logger.info(f"explicit feedback logging with {inference_samples} requests .")

    wos_client.data_sets.store_records(
        data_set_id=feedback_data_set_id,
        request_body=[scoring_feedback_wos],
        background_mode=False,
    )

    fl_records_after = wos_client.data_sets.get_records_count(feedback_data_set_id)

    logger.info(
        f"{fl_records_before} records in the feedback logging table before adding {inference_samples}"
    )
    logger.info(
        f"{fl_records_after} records in the feedback logging table after adding {inference_samples}"
    )

    return {
        "payload_dataset_id": feedback_data_set_id,
        "payload_sent": inference_samples,
        "payload_processed": fl_records_after,
        "success": fl_records_after - fl_records_before == inference_samples,
    }
