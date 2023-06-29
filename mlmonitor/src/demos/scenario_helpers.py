# SPDX-License-Identifier: Apache-2.0
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from typing import Callable
import random
import time
import uuid
import pandas as pd
import numpy as np
from ibm_watson_openscale.supporting_classes.enums import DataSetTypes, TargetTypes
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord

from mlmonitor.src.wos import wos_client
from mlmonitor.src import logger
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment
from mlmonitor.src.model.config import ModelConfig

dd_scenario1 = [
    0.01,
    0.02,
    0.08,
    0.10,
    0.11,
    0.12,
    0.12,
    0.13,
    0.13,
    0.14,
    0.14,
    0.15,
    0.16,
]


def log_perturbed_data(
    model_config: ModelConfig,
    deployment_name: str,
    rand: bool,
    deployment_target: str = "aws",
    perturb: bool = False,
    perturb_args: dict = None,
    inference_samples: int = 2,
    inference_list: list = None,
) -> None:
    """send <inference_samples> Payload logging requests to the deployed Sagemaker Endpoint with name <deployment_name>
    Parameters
    ----------
    model_config : str
       this indicates the location of source code and dependencies to be uploaded and used by the endpoint
    deployment_name : str
       this indicates the Endpoint Name in Sagemaker for which an OpenScale subscription should be created
    deployment_target : str
       target model serving environment
       wml , aws , azure , custom
    rand : bool
        Whether to send random samples to payload logging. If False, will use the samples provided in inference_list
    perturb : bool
        If true will perturb a column based on arguments given in perturb_args
    perturb_args : dict
        Arguments to send to the perturb_column function.
        Must contain 3 entries:
            col: name of column to change
            ratio: Number between 0 and 1. Percentage of rows to apply the operation to
            operation: function to apply to each row of the specified column
    inference_samples : int
       Number of inference samples to send in the scoring request to the deployed Endpoint (deployment_name)
    inference_list : list
        If specifided, will use values from this list as the samples to send to payload logging. Can be used to specify exact data records to send instead of random records.
    data_path : str
       location of dataset to be fetched to get scoring request samples
    Returns
    -------
    """

    assert deployment_target in {"wml", "aws"}, "Deployment target should be wml or aws"

    ###########################
    # VERIFY WOS SUBSCRIPTION #
    ###########################

    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=deployment_name
    )

    if len(subscription_ids) != 1:
        raise ValueError(
            f"{deployment_name} should have exactly one subscription ID => {len(subscription_ids)} found"
        )

    subscription_id = subscription_ids[0]

    logger.debug(
        f"Deployment Name {deployment_name}\n" f"subscription_id {subscription_id}\n"
    )

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

    logger.info(
        f'Found PAYLOAD_DATASET_ID for SUBSCRIPTION_ID  "{subscription_id}" : {payload_dataset_id}'
    )

    pl_records_before = wos_client.data_sets.get_records_count(payload_dataset_id)

    ###############################
    # FETCH NON PERTURBED DATASET #
    ###############################

    assert (
        model_config.data_type == "structured"
    ), ' "log_payload_data" only supported for structured data types and csv format'

    df = model_config._get_data(num_samples=inference_samples)
    df = df.loc[:, model_config.feature_columns]

    ###########################
    # PERTURB DATASET         #
    ###########################

    if perturb:
        assert perturb_args is not None, "Arguments must be defined for perturbation"

        scoring_df = perturb_column(
            df,
            col=perturb_args.get("col"),
            ratio=perturb_args.get("ratio"),
            operation=perturb_args.get("operation"),
            total_records=inference_samples,
        )

    elif rand:
        # send random samples
        scoring_df = df.iloc[random.sample(range(df.shape[0]), inference_samples), :]
    else:
        # send specific samples from the dataset
        scoring_df = df.iloc[inference_list, :]

    #################################
    #  PREPARE SCORING PAYLOAD      #
    #################################

    fields = scoring_df.columns.to_list()
    values = scoring_df.values
    int_cols = [
        col for col, dtype in scoring_df.dtypes.to_dict().items() if dtype == np.int64
    ]
    to_int = lambda values: [
        int(v) if f in int_cols else v for f, v in zip(fields, values)
    ]

    scoring_payload_wos = {
        "fields": fields,
        "values": [to_int(value) for value in values],
    }

    #################################
    #  GET SCORING RESPONSE         #
    #################################

    wos_response_params = {"df": scoring_df, "endpoint_name": deployment_name}

    if deployment_target == "aws":

        from mlmonitor.src import aws_credentials
        from mlmonitor.src.aws.scoring import get_wos_response

        wos_response_params = {
            **wos_response_params,
            **aws_credentials,
            "prediction_field": model_config.prediction_field,
            "probability_field": model_config.probability_fields[0],
        }

    elif deployment_target == "wml":

        from mlmonitor.src.wml import WML_SPACE_ID, wml_client as WML_CLIENT
        from mlmonitor.src.wml.scoring import get_wos_response

        WML_CLIENT.set.default_space(WML_SPACE_ID)

        wos_response_params["wml_client"] = WML_CLIENT
    else:
        raise NotImplementedError("deployment_target should be aws or wml")

    # scoring_prediction => implicit payload logging for structured
    scoring_response_wos = get_wos_response(**wos_response_params)

    ########################################################
    #  RUN EXPLICIT  PAYLOAD LOGGING FOR NON WML MODELS    #
    ########################################################

    if deployment_target != "wml":
        logger.info(
            f"explicit payload logging for non-WML model serving environment with {inference_samples} requests."
        )

        logger.info(f"explicit payload logging with {scoring_df.shape[0]} requests .")
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


def log_feedback_data(
    deployment_name: str,
    model_config: ModelConfig,
    aws_credentials: dict,
    rand: bool = True,
    inference_samples: int = 2,
    inference_list: list = None,
    include_predictions: bool = False,
) -> None:
    """send <inference_samples> Feedback logging requests to the deployed Sagemaker Endpoint with name <deployment_name>
    Parameters
    ----------
    model_config : str
        this indicates the location of source code and dependencies to be uploaded and used by the endpoint
    deployment_name : str
       this indicates the Endpoint Name in Sagemaker for which an OpenScale subscription should be created
    aws_credentials : dict
       AWS credentials to use to invoke deployed online inference endpoint in AWS Sagemaker. this dictionary should
       be formatted as follow:
        {"access_key": <your access_key>,
        "secret_key": <your secret>,
        "region_name": <your region name>}
    rand : bool
        Whether or not to send random samples to payload logging. If False, will use the samples provided in inference_list
    inference_samples : int
       Number of inference samples to send in the scoring request to the deployed Endpoint (deployment_name)
    inference_list : list
        If specifided, will use values from this list as the samples to send to payload logging. Can be used to specify exact data records to send instead of random records.
    include_predictions : bool
       This boolean flag indicates whether to include model predictions to the feedback logging request or
       to let OpenScale query the online endpoint to obtain predictions
    data_path : str
       location of dataset to be fetched to get scoring request samples
    file_name : str
        name of data file to read
    -------
    """
    from mlmonitor.src.aws.scoring import get_scores_labels

    assert (
        model_config.data_type == "structured"
    ), ' "log_payload_data" only supported for structured data types and csv format'

    df = model_config._get_data(num_samples=inference_samples)

    if rand:
        mask = random.sample(range(df.shape[0]), inference_samples)
    else:
        mask = inference_list
    print(mask)
    request_df = df.loc[:, model_config.feature_columns]
    scoring_df = df.loc[:, model_config.class_label]

    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=deployment_name
    )

    if len(subscription_ids) != 1:
        raise ValueError(
            f"{deployment_name} should have exactly one subscription ID => {len(subscription_ids)} found"
        )

    subscription_id = subscription_ids[0]

    logger.debug(
        f"Deployment Name {deployment_name}\n" f"subscription_id {subscription_id}\n"
    )

    # Datasets PAYLOAD_DATASET
    feedback_data_set_id = (
        wos_client.data_sets.list(
            type=DataSetTypes.FEEDBACK,
            target_target_id=subscription_id,
            target_target_type=TargetTypes.SUBSCRIPTION,
        )
        .result.data_sets[0]
        .metadata.id
    )

    logger.info(
        f'Found feedback_data_set_id for SUBSCRIPTION_ID  "{subscription_id}" : {feedback_data_set_id}'
    )

    if feedback_data_set_id is None:
        raise ValueError(
            "Feedback data set not found. Please check subscription status."
        )
    else:
        logger.info(f"Feedback data set id:{feedback_data_set_id}")

    # Feedback request with predictions
    proba_vectors, proba_scores, predicted_labels = get_scores_labels(
        df=scoring_df,
        aws_access_key_id=aws_credentials.get("access_key"),
        aws_secret_access_key=aws_credentials.get("secret_key"),
        region_name=aws_credentials.get("region_name"),
        endpoint_name=deployment_name,
    )

    scoring_feedback_wos1 = {
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

    # Feedback request without predictions
    scoring_feedback_wos2 = {
        "fields": request_df.columns.tolist(),
        "values": request_df.values.tolist(),
    }

    fl_records_before = wos_client.data_sets.get_records_count(feedback_data_set_id)

    logger.info(f"explicit feedback logging with {scoring_df.shape[0]} requests .")
    request_body = (
        [scoring_feedback_wos1] if include_predictions else [scoring_feedback_wos2]
    )
    wos_client.data_sets.store_records(
        data_set_id=feedback_data_set_id,
        request_body=request_body,
        background_mode=False,
    )

    fl_records_after = wos_client.data_sets.get_records_count(feedback_data_set_id)

    logger.debug(
        f"{fl_records_before} records in the feedback logging table before adding {inference_samples}"
    )
    logger.debug(
        f"{fl_records_after} records in the feedback logging table after adding {inference_samples}"
    )


def extract_performance_from_response(model_responses, request_df):
    y_pred = [res.get("predicted_label") for res in model_responses]
    y_true = request_df["Churn?_True."].tolist()
    return y_pred, y_true


def get_confusion_matrix_from_response(model_responses, request_df):
    y_pred, y_true = extract_performance_from_response(model_responses, request_df)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return tn, fp, fn, tp


def get_confusion_matrix_records_from_response(model_responses, request_df):
    y_pred, y_true = extract_performance_from_response(model_responses, request_df)

    assert len(y_pred) == len(
        y_true
    ), "Length of predicted values and true values must be equal"

    tn_mask = []
    fp_mask = []
    fn_mask = []
    tp_mask = []

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_pred[i] == 1:
                tp_mask.append(i)
            else:
                tn_mask.append(i)

        elif y_pred[i] == 1:
            fp_mask.append(i)
        else:
            fn_mask.append(i)
    return tn_mask, fp_mask, fn_mask, tp_mask


def print_scikit_metrics(model_responses, request_df):
    y_pred, y_true = extract_performance_from_response(model_responses, request_df)

    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred)}")
    print(f"Recall: {recall_score(y_true, y_pred)}")
    print(f"ROC: {roc_auc_score(y_true, y_pred)}")


def get_all_scenarios(tn_mask, fp_mask, fn_mask, tp_mask, metric, target_metric_list):
    assert metric in [
        "accuracy",
        "precision",
        "recall",
    ], "metric parameter must be one of: 'accuracy', 'precision', 'recall'"

    all_scenarios = []

    if metric == "accuracy":
        for acc in target_metric_list:
            wrong_pred = fp_mask + fn_mask
            correct_pred = tp_mask + tn_mask

            preds = []
            preds.extend(random.sample(correct_pred, acc))
            preds.extend(random.sample(wrong_pred, 100 - acc))

            all_scenarios.append(preds)
    elif metric == "precision":
        # precision= tp/(tp + fp)
        for prec in target_metric_list:
            preds = []

            preds.extend(random.sample(tp_mask, prec))
            preds.extend(random.sample(fp_mask, 100 - prec))
            # preds.extend(random.sample(tn_mask, 10))
            # preds.extend(random.sample(fn_mask, 10))

            all_scenarios.append(preds)
    elif metric == "recall":
        # recall= tp/(tp + fn)
        for rec in target_metric_list:
            preds = []

            preds.extend(random.sample(tp_mask, rec))
            preds.extend(random.sample(fn_mask, 100 - rec))
            # preds.extend(random.sample(tn_mask, 10))
            # preds.extend(random.sample(fp_mask, 10))

            all_scenarios.append(preds)

    return all_scenarios


def perturb_column(
    df: pd.DataFrame,
    total_records: int,
    ratio: float,
    target_column: str,
    perturbation_fn: Callable,
) -> pd.DataFrame:
    """
    Takes a dataframe, column name, ratio of records to perturb
    (0.2 means 20% of the records will be perturbed), and an operation to
    perform on the column.It returns a new dataframe with the specified
    number of records randomly selected from the original dataframe having
    their specified columns changed according to the operation.
    :param df: Specify the dataframe that is being perturbed
    :param col:str: Specify the column that we want to perturb
    :param ratio:float: Determine the ratio of records that will be perturbed
    :param operation:Callable Specify the operation that is performed on the column
    :param total_records:int Specify the number of records that will be used to generate the perturbations
    :return: A dataframe with the same number of rows as the original dataframe, but with a perturbed version of one column
    """

    num_perturb = int(ratio * total_records)
    mask = random.sample(df.index.tolist(), num_perturb)
    logger.info(f"{num_perturb}/{len(df)} perturbed samples")

    df_payload = df.copy()
    df_payload.loc[mask, target_column] = df_payload.loc[mask, target_column].apply(
        lambda x: eval(perturbation_fn)
    )

    return df_payload


def perturb_double_column(
    df: pd.DataFrame,
    total_records: int,
    ratio: float,
    source_column: str,
    source_cond: str,
    target_column: str,
    perturbation_fn: Callable,
) -> pd.DataFrame:

    # Randomly sampling the DataFrame
    payload_df = df.sample(total_records)

    # From the sample, nb of records with respect to the condition
    cond = payload_df[source_column] == source_cond
    cond_records = len(payload_df.loc[cond])

    # Defining number of perturbations based on the the records meeting the condition
    num_perturb = int(np.ceil(ratio * cond_records))
    perturbation_mask = random.sample(payload_df.loc[cond].index.tolist(), num_perturb)

    # target_mean = payload_df[target_column].mean()

    payload_df.loc[perturbation_mask, target_column] = payload_df.loc[
        perturbation_mask, target_column
    ].apply(
        lambda x: eval(perturbation_fn)
        # lambda x: eval("x + target_mean*3")
    )

    return payload_df
