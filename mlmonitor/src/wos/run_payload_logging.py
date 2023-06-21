# SPDX-License-Identifier: Apache-2.0
import time
import uuid
from typing import Callable, Dict

from ibm_watson_openscale.supporting_classes.enums import DataSetTypes, TargetTypes
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord

from mlmonitor.src import logger
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.model.config import ModelConfig


def log_as_wos_payload(deployment_target: str) -> Callable:
    def decorator(function_task):
        def log_wrapper(*args, **kwargs):
            self = args[0]
            dataset_type = args[1]
            inference_samples = kwargs.get("total_records")
            assert deployment_target in {
                "wml",
                "aws",
                "azure",
            }, "Deployment target should be wml aws or azure"
            assert dataset_type in {"train", "test", "validation"}

            ######################################################################
            #  Datasets PAYLOAD_DATASET ID - COUNT RECORDS                       #
            ######################################################################
            payload_dataset_id = (
                wos_client.data_sets.list(
                    type=DataSetTypes.PAYLOAD_LOGGING,
                    target_target_id=self.subscription_id,
                    target_target_type=TargetTypes.SUBSCRIPTION,
                )
                .result.data_sets[0]
                .metadata.id
            )

            assert payload_dataset_id

            pl_records_before = wos_client.data_sets.get_records_count(
                payload_dataset_id
            )

            assert (
                self._model_config.data_type == "structured"
            ), "log_payload_data only supported for structured data types"

            ###########################
            # PERTURB DATASET         #
            ###########################
            df = self._model_config._get_data(
                num_samples=inference_samples, dataset_type=dataset_type
            )
            inference_samples = min(inference_samples, df.shape[0])
            # assign scoring transaction_id
            # df['transaction_id'] = df.apply(lambda x : f"{str(uuid.uuid4()).replace('-','')}-{str(x.name)}",axis=1)
            self._df = df.loc[:, self._model_config.feature_columns]
            scoring_df = function_task(self, **kwargs)
            #################################
            #  PREPARE SCORING PAYLOAD      #
            #################################
            scoring_payload_wos = {
                "fields": scoring_df.columns.to_list(),
                "values": [
                    list(x.values())
                    for x in scoring_df.to_dict(orient="index").values()
                ],
            }
            #################################
            #  GET SCORING RESPONSE         #
            #################################

            wos_response_params = {
                "df": scoring_df,
                "endpoint_name": self.model_endpoint,
            }

            if deployment_target == "aws":

                from mlmonitor.src import aws_credentials
                from mlmonitor.src.aws.scoring import get_wos_response

                wos_response_params = {
                    **wos_response_params,
                    **aws_credentials,
                    "prediction_field": self._model_config.prediction_field,
                    "probability_field": self._model_config.probability_fields[0],
                }

            elif deployment_target == "azure":

                from mlmonitor.src.azure import AZ_WORKSPACE
                from mlmonitor.src.azure.scoring import get_wos_response

                wos_response_params = {
                    **wos_response_params,
                    "workspace": AZ_WORKSPACE,
                    "prediction_field": self._model_config.prediction_field,
                    "probability_field": self._model_config.probability_fields[0],
                }

            elif deployment_target == "wml":

                from mlmonitor.src.wml import WML_SPACE_ID, wml_client as WML_CLIENT
                from mlmonitor.src.wml.scoring import get_wos_response

                WML_CLIENT.set.default_space(WML_SPACE_ID)

                wos_response_params["wml_client"] = WML_CLIENT
            else:
                raise NotImplementedError(
                    "deployment_target should be aws wml or azure"
                )

            # scoring_prediction => implicit payload logging for structured
            scoring_response_wos = get_wos_response(**wos_response_params)

            ########################################################
            #  RUN EXPLICIT  PAYLOAD LOGGING FOR NON WML MODELS    #
            ########################################################

            if deployment_target != "wml":
                logger.info(
                    f"explicit payload logging for non-WML model serving environment with {inference_samples} requests."
                )

                logger.info(
                    f"explicit payload logging with {scoring_df.shape[0]} requests ."
                )
                wos_client.data_sets.store_records(
                    data_set_id=payload_dataset_id,
                    background_mode=False,
                    request_body=[
                        PayloadRecord(
                            scoring_id=str(uuid.uuid4()),
                            request=scoring_payload_wos,
                            response=scoring_response_wos,
                            response_time=460,
                        )
                    ],
                )

            ######################################################################
            #  Datasets PAYLOAD_DATASET ID - COUNT RECORDS                       #
            ######################################################################
            pl_records_after = wos_client.data_sets.get_records_count(
                payload_dataset_id
            )
            patience = 12
            while (
                pl_records_after - pl_records_before
            ) != inference_samples and patience >= 0:
                logger.info(
                    f"wait 5s {pl_records_after} payload records , expecting {pl_records_before + inference_samples} "
                )
                time.sleep(5)
                pl_records_after = wos_client.data_sets.get_records_count(
                    payload_dataset_id
                )
                patience -= 1

            logger.info(
                f"{pl_records_before} records in the payload logging table before adding {inference_samples}"
            )
            logger.info(
                f"{pl_records_after} records in the payload logging table after adding {inference_samples}"
            )

        return log_wrapper

    return decorator


def log_payload_data(
    model_config: ModelConfig,
    deployment_name: str,
    deployment_target: str = "aws",
    inference_samples: int = 2,
    dataset_type: str = "validation",
) -> Dict:
    """
    send <inference_samples> Payload logging requests to the deployed Sagemaker,Azure or WML Endpoint with name <deployment_name>

    :param model_config: ModelConfig: model configuration
    :param deployment_name: str: Endpoint Name in Sagemaker Azure or WML for which an OpenScale subscription should be created
    :param deployment_target: str: target model serving environment wml , aws , azure
    :param inference_samples: int: Number of inference samples to send in the scoring request to the deployed Endpoint (deployment_name)
    :param dataset_type: str: dataset type to use for payload logging test , train , validation
    :return: Dict dictionary with payload_dataset_id,payload_sent,payload_processed,success keys
    """
    assert deployment_target in {
        "wml",
        "aws",
        "azure",
    }, "Deployment target should be wml ,aws or azure"
    assert dataset_type in {"test", "train", "validation"}

    ###########################
    # VERIFY WOS SUBSCRIPTION #
    ###########################
    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=deployment_name
    )

    if len(subscription_ids) != 1:
        raise ValueError(
            f"{deployment_name} has {len(subscription_ids)} subscription(s) , 1 expected"
        )

    subscription_id = subscription_ids[0]
    logger.debug(
        f"Deployment Name {deployment_name}\n" f"subscription_id {subscription_id}\n"
    )

    ######################################################################
    # RETRIEVE MODEL USE CASE SPECIFIC CONFIGURATION DETAILS (SIGNATURE) #
    ######################################################################

    logger.info(f"Payload Logging for {deployment_name} in {deployment_target}")

    if deployment_target == "aws":
        from mlmonitor.src import aws_credentials
        from mlmonitor.src.aws.scoring import get_wos_response, _score_unstructured
    elif deployment_target == "wml":
        from mlmonitor.src.wml import WML_SPACE_ID, wml_client as WML_CLIENT
        from mlmonitor.src.wml.scoring import get_wos_response, _score_unstructured

        WML_CLIENT.set.default_space(WML_SPACE_ID)
    elif deployment_target == "azure":
        from mlmonitor.src.azure import AZ_WORKSPACE
        from mlmonitor.src.azure.scoring import get_wos_response, _score_unstructured
    else:
        raise NotImplementedError("deployment_target should be aws wml or azure")
    ######################################################################
    #  Datasets PAYLOAD_DATASET ID - COUNT RECORDS                       #
    ######################################################################
    payload_dataset_id = (
        wos_client.data_sets.list(
            type=DataSetTypes.PAYLOAD_LOGGING,
            target_target_id=subscription_id,
            target_target_type=TargetTypes.SUBSCRIPTION,
        )
        .result.data_sets[0]
        .metadata.id
    )

    assert payload_dataset_id

    logger.info(f"Payload data set id: {payload_dataset_id}")

    pl_records_before = wos_client.data_sets.get_records_count(payload_dataset_id)

    #########################################################################
    #  SCORING REQUEST + SCORING RESPONSE - STRUCTURED AND UNSTRUCTURED      #
    #########################################################################
    dataset = model_config._get_data(
        num_samples=inference_samples, dataset_type=dataset_type
    )

    if model_config.data_type == "structured":

        inference_samples = min(inference_samples, dataset.shape[0])
        #################################
        #  PREPARE SCORING PAYLOAD      #
        #################################
        scoring_df = dataset.loc[:, model_config.feature_columns]

        scoring_payload_wos = {
            "fields": scoring_df.columns.to_list(),
            "values": [
                list(x.values()) for x in scoring_df.to_dict(orient="index").values()
            ],
        }

        #################################
        #  GET SCORING RESPONSE         #
        #################################

        wos_response_params = {"df": scoring_df, "endpoint_name": deployment_name}

        if deployment_target == "aws":
            wos_response_params = {
                **wos_response_params,
                **aws_credentials,
                "prediction_field": model_config.prediction_field,
                "probability_field": model_config.probability_fields[0],
            }
        elif deployment_target == "azure":
            wos_response_params = {
                **wos_response_params,
                "workspace": AZ_WORKSPACE,
                "prediction_field": model_config.prediction_field,
                "probability_field": model_config.probability_fields[0],
            }
        elif deployment_target == "wml":
            wos_response_params["wml_client"] = WML_CLIENT
        else:
            raise NotImplementedError("deployment_target should be aws or wml")

        # scoring_prediction => implicit payload logging for structured
        scoring_response_wos = get_wos_response(**wos_response_params)

    elif model_config.data_type == "unstructured_image":

        #################################
        #  PREPARE SCORING PAYLOAD      #
        #################################

        samples, labels = dataset

        scoring_payload_wos = {"values": samples.tolist()}

        #################################
        #  GET SCORING RESPONSE         #
        #################################

        if deployment_target in {"aws", "azure"}:
            scoring_response_wos = _score_unstructured(
                payload=samples,
                endpoint_name=deployment_name,
                **aws_credentials,
            )
        elif deployment_target == "wml":
            # scoring_prediction => implicit payload logging for unstructured
            scoring_response_wos = _score_unstructured(
                payload=samples, endpoint_name=deployment_name, client=WML_CLIENT
            )
        else:
            raise NotImplementedError("deployment_target should be aws or wml")

        scoring_response_wos = scoring_response_wos.get("predictions")[0]

    else:
        raise ValueError("data_type should be structured (csv) or unstructured_image ")

    logger.debug(f"openscale payload request : {scoring_payload_wos}")
    logger.debug(f"openscale payload response: {scoring_response_wos}")

    ########################################################
    #  RUN EXPLICIT  PAYLOAD LOGGING FOR NON WML MODELS    #
    ########################################################

    if deployment_target != "wml":
        logger.info(
            f"explicit payload logging for non-WML model serving environment with {inference_samples} requests."
        )
        wos_client.data_sets.store_records(
            data_set_id=payload_dataset_id,
            background_mode=False,
            request_body=[
                PayloadRecord(
                    scoring_id=str(uuid.uuid4()),
                    request=scoring_payload_wos,  # {"fields":[],values:[]}
                    response=scoring_response_wos,  # {"fields":[],values:[]}
                    response_time=460,
                )
            ],
        )

    ######################################################################
    #  Datasets PAYLOAD_DATASET ID - COUNT RECORDS                       #
    ######################################################################
    pl_records_after = wos_client.data_sets.get_records_count(payload_dataset_id)
    patience = 12
    while (pl_records_after - pl_records_before) != inference_samples and patience >= 0:
        logger.info(
            f"wait 5s {pl_records_after} payload records , expecting {pl_records_before + inference_samples} "
        )
        time.sleep(5)
        pl_records_after = wos_client.data_sets.get_records_count(payload_dataset_id)
        patience -= 1

    logger.info(
        f"{pl_records_before} records in the payload logging table before adding {inference_samples}"
    )
    logger.info(
        f"{pl_records_after} records in the payload logging table after adding {inference_samples}"
    )

    return {
        "payload_dataset_id": payload_dataset_id,
        "payload_sent": inference_samples,
        "payload_processed": pl_records_after,
        "success": pl_records_after - pl_records_before == inference_samples,
    }
