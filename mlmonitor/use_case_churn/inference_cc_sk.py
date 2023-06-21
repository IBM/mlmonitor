# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np

# from sagemaker_inference import decoder
import pickle as pkl
import logging
import uuid
import pandas as pd
import json
from ibm_watson_openscale.supporting_classes.enums import TargetTypes, DataSetTypes
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord
from ibm_watson_openscale import APIClient as WOS_APIClient
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

try:
    from utils import _get_secret, read_columns
except ImportError as e:
    print(
        f"use_case_churn.inference_cc_sk could not import modules => not running in AWS Endpoint : {e}"
    )
    from mlmonitor.use_case_churn.utils import _get_secret, read_columns

SUBSCRIPTION_ID = None
FEATURES = read_columns()


# INFERENCE IN SAGEMAKER
def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    print("model_fn model_fn model_fn model_fn ")
    model_file = "xgboost-model-sk"
    model = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return model


def input_fn(request_body, request_content_type):
    """An input_fn that loads a csv"""
    log.info(f"Called input_fn {request_content_type}")
    log.info(request_body)
    if request_content_type == "text/csv":
        # data = StringIO(request_body)
        request_body = request_body.lstrip('"').rstrip('"')
        inputs = np.array(
            [[float(y) for y in row.split(",")] for row in request_body.split("\n")]
        )
        return inputs
    elif request_content_type == "application/json":
        jsondata = json.loads(request_body)

        global SUBSCRIPTION_ID
        SUBSCRIPTION_ID = jsondata.get("subscription_id")
        # ! TODO send integers for OHE features instead of float
        values = [
            [float(y) for y in x.get("features")] for x in jsondata.get("instances")
        ]
        if SUBSCRIPTION_ID:
            log.info(f"subscription ID passed for payload logging {SUBSCRIPTION_ID}")
            global scoring_payload_wos
            scoring_payload_wos = {"fields": FEATURES, "values": values}
            log.info(f"wos inputs {scoring_payload_wos}")
        return np.array(values)
    # else:
    # Handle other content-types here or raise an Exception
    # if the content type is not supported.
    # np_array = decoder.decode(request_body, request_content_type)
    # return np_array


# INFERENCE IN OPENSHIFT
def input_fn_ocp(jsondata, request_content_type):
    """An input_fn that loads json"""
    log.info(f"Called input_fn_ocp {request_content_type}")
    log.info(jsondata)

    if request_content_type != "application/json":
        raise ValueError("Content Type should be application/json")
    # fields values format (WOS)
    if "values" in jsondata.keys():

        FIELDS = jsondata.get("fields")
        all_features = jsondata.get("values")
        if FIELDS != FEATURES:
            raise ValueError(f"passed values should be \n{FEATURES}\n{FIELDS} received")

        values = [[float(val) for val in values] for values in all_features]

    # AWS format instances featuresins
    elif "instances" in jsondata.keys():
        all_features = jsondata.get("instances")

        values = [[float(y) for y in x.get("features")] for x in all_features]

    else:
        raise ValueError("Unsupported input json format received")

    return np.array(values)


# INFERENCE IN OPENSHIFT
def output_fn_ocp(predictions, fields, values):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """

    predicted_labels = [int(np.argmax(probas)) for probas in predictions.tolist()]

    proba_scores = [round(max(probas), 3) for probas in predictions.tolist()]

    proba_vectors = [
        list(np.roll(np.array([np.round(proba, 3), np.round(1 - proba, 3)]), label))
        for proba, label in zip(proba_scores, predicted_labels)
    ]
    scoring_response_wos = {
        "fields": fields + ["predicted_label", "score", "prediction_probability"],
        "values": [
            w + [x] + [y] + [z]
            for w, x, y, z in zip(values, predicted_labels, proba_scores, proba_vectors)
        ],
    }

    return scoring_response_wos


# INFERENCE IN SAGEMAKER
def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.
    Return a two-dimensional NumPy array where the first columns are predictions
    and the remaining columns are the feature contributions (SHAP values) for that prediction.
    """
    return model.predict_proba(input_data)


# INFERENCE IN SAGEMAKER
def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    log.info(f"output_fn received {predictions}")
    if content_type == "text/csv":
        res = ",".join(str(x) for x in predictions)
        log.info(f"output_fn return {content_type} response :\n{res}")
        return ",".join(str(x) for x in predictions)

    if content_type != "application/json":
        raise ValueError(f"Content type {content_type} is not supported.")
    outputs = [
        {"predicted_label": int(np.argmax(probas)), "score": round(max(probas), 3)}
        for probas in predictions.tolist()
    ]
    res = {"predictions": outputs}
    log.info(f"output_fn return {content_type} response :\n{res}")
    if SUBSCRIPTION_ID:
        _payload_logging_wos(predictions)
    return {"predictions": outputs}


def _payload_logging_wos(predictions):
    """
    perform payload logging to Watson OpenScale subscription identified by global variable SUBSCRIPTION_ID from `input_fn`
    It includes :
        1) model `predictions` input parameter transformed into a dictionary compliant to WOS format
        2) scoring inputs global variables `scoring_payload_wos` from `input_fn` already compliant to WOS format

    A WOS compliant format has `fields` and `values` keys.

    :param predictions: predictions received by Endpoint
    :return: The scoring_response_wos dictionary
    """
    API_KEY = json.loads(_get_secret(secret_name="IBM_KEYS")).get("IBM_API_KEY_MLOPS")
    authenticator = IAMAuthenticator(apikey=API_KEY)
    wos_client = WOS_APIClient(authenticator=authenticator)

    payload_data_set_id = (
        wos_client.data_sets.list(
            type=DataSetTypes.PAYLOAD_LOGGING,
            target_target_id=SUBSCRIPTION_ID,
            target_target_type=TargetTypes.SUBSCRIPTION,
        )
        .result.data_sets[0]
        .metadata.id
    )

    scoring_response_wos = {
        "fields": ["predicted_label", "score"],
        "values": [
            [int(np.argmax(probas)), round(max(probas), 3)]
            for probas in predictions.tolist()
        ],
    }
    # write predictions
    log.info(f"wos outputs {scoring_response_wos}")
    wos_client.data_sets.store_records(
        data_set_id=payload_data_set_id,
        background_mode=True,
        request_body=[
            PayloadRecord(
                scoring_id=str(uuid.uuid4()),
                request=scoring_payload_wos,
                response=scoring_response_wos,
                response_time=460,
            )
        ],
    )
    log.info("payload logging completed")


# INFERENCE IN AZURE
def init():
    # TODO pick model name from Container environment variables set to "model.joblib"
    global model
    print(f"loading model form {os.getenv('AZUREML_MODEL_DIR')}")
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model = model_fn(model_dir=model_dir)


def run(input_data):
    print("start inference")
    try:

        dict_data = input_data
        print(f"input data (json):\n{dict_data}")

        input_data = pd.DataFrame.from_dict(dict_data["input"])

        # load
        predictions = model.predict_proba(input_data)
        scores = model.predict_proba(input_data)

        records = [
            {"Scored Labels": int(pred), "Scored Probabilities": prob}
            for pred, prob in zip(predictions, scores)
        ]
        result = {"output": records}
        print(f"output:data:\n{result}")

        return result
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return {"error": result}
