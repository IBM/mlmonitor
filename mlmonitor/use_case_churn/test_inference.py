# SPDX-License-Identifier: Apache-2.0
import os
import json
import pandas as pd
from os.path import dirname
from typing import Callable, Union, Dict
from use_case_churn.inference_cc_xg_boost import (
    model_fn,
    input_fn,
    predict_fn,
    output_fn,
)

from use_case_churn.inference_cc_sk import (
    model_fn as model_fn_sk,
    input_fn as input_fn_sk,
    predict_fn as predict_fn_sk,
    output_fn as output_fn_sk,
)
from use_case_churn.test_train import Env
from use_case_churn.utils import read_columns

PROJECT_ROOT = dirname(dirname(__file__))


def inference(
    mdlf: Callable,
    inputf: Callable,
    predf: Callable,
    outpf: Callable,
    content_type: str,
    payload: Union[Dict, str],
):
    """
    takes a payload and inference functions required by Sagemaker inference

    https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html
    https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html

    :param mdlf: model_fn function to use to Load a Model
    :param inputf: input_fn function to process model inputs return an object that can be passed to predict_fn
    :param predf: predict_fn  should invoke the loaded model and call the predict function on input_object, and return the resulting value.
    :param outpf: output_fn Process outputs
    :param content_type: Specify the content type of the input data
    :param payload: Pass the data from the http request to the inference function json or csv in text
    :return: predictions in json format
    """
    booster = mdlf(os.environ["SM_MODEL_DIR"])
    input_payload = inputf(payload, content_type)
    outputs = predf(input_payload, booster)
    return outpf(outputs, content_type)


if __name__ == "__main__":
    Env()
    # Generate Scoring Inputs
    inference_samples = 2
    content_type = "text/csv"
    test_data = pd.read_csv(
        os.path.join(PROJECT_ROOT, "datasets", "churn", "validation.csv"),
        engine="python",
        header=None,
    )
    subscription_id = "c621d686-5e36-4bfc-8ac4-271cc7ee6074"
    input_data = test_data.iloc[1:6, 1:].values

    npy_payload = test_data.to_numpy()[1 : inference_samples + 1, 1:]
    csv_payload = "\n".join([",".join([str(x) for x in row]) for row in npy_payload])
    # csv_payload = '\"'+csv_payload+'\"'
    json_payload = {"instances": [{"features": x} for x in npy_payload.tolist()]}

    #  Test Inference code for Scikit Learn Sagemaker Model handling csv input format
    preds_sk_csv = inference(
        model_fn_sk, input_fn_sk, predict_fn_sk, output_fn_sk, "text/csv", csv_payload
    )
    print("preds_sk_csv", preds_sk_csv)
    preds_sk_json = inference(
        model_fn_sk,
        input_fn_sk,
        predict_fn_sk,
        output_fn_sk,
        "application/json",
        json.dumps(json_payload),
    )

    print("preds_sk_json", preds_sk_json)

    print("predictions with OpenScale")
    # json_payload["subscription_id"] = subscription_id
    #  Test Inference code for Scikit Learn Sagemaker Model handling WOS input format
    preds_sk_json_wos = inference(
        model_fn_sk,
        input_fn_sk,
        predict_fn_sk,
        output_fn_sk,
        "application/json",
        json.dumps(json_payload),
    )
    print("preds_sk_json_wos", preds_sk_json_wos)
    #  Test Inference code for XGBOOST Sagemaker Model handling csv format
    preds_xgb_csv = inference(
        model_fn, input_fn, predict_fn, output_fn, "text/csv", csv_payload
    )
    print("preds_xgb_csv", preds_xgb_csv)

    # Test Inference code for WOS Custom ML provider (REST endpoint deployed in Openshift)
    print("predictions with OCP")
    FEATURES = read_columns()
    test_data = pd.read_csv(
        os.path.join(PROJECT_ROOT, "datasets", "churn", "validation.csv"),
        engine="python",
    )
    npy_payload = test_data.loc[1 : inference_samples + 1, FEATURES].to_numpy()
    from use_case_churn.inference_cc_sk import input_fn_ocp, output_fn_ocp

    json_payload = {"values": npy_payload.tolist(), "fields": FEATURES}

    booster = model_fn_sk(os.environ["SM_MODEL_DIR"])
    input_payload = input_fn_ocp(json_payload, "application/json")
    outputs = predict_fn_sk(input_payload, booster)
    preds_sk_json_ocp = output_fn_ocp(
        outputs,
        FEATURES,
        test_data.loc[1 : inference_samples + 1, FEATURES].values.tolist(),
    )

    print("preds_sk_json_ocp", preds_sk_json_wos)
