# SPDX-License-Identifier: Apache-2.0
import joblib
import os
import pandas as pd
import logging
import json
from io import StringIO
from utils import read_columns

try:
    from sagemaker_inference import decoder

except ModuleNotFoundError as e:
    print(f"running locally : {e}")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def predict_fn(input_data, model):
    log.info("Called predict_fn ")
    log.info(input_data)
    COLUMNS = read_columns()
    df = pd.DataFrame(input_data, columns=COLUMNS)
    pred = model.predict(df)
    scores = model.predict_proba(df)
    records = [
        {
            "predicted_label": int(pred),
            "score": prob[pred],
        }
        for pred, prob in zip(pred, scores)
    ]
    log.info("predictions")
    log.info(records)
    return records


def input_fn(request_body, request_content_type):
    """An input_fn that loads a csv"""
    log.info("Called input_fn " + request_content_type)
    log.info(request_body)
    if request_content_type == "text/csv":
        COLUMNS = read_columns()
        data = StringIO(request_body)
        df = pd.read_csv(data, sep=",", header=None, names=COLUMNS)
        log.info("returning input for prediction")
        return df.to_numpy()
    elif request_content_type == "application/json":
        jsondata = json.loads(request_body)
        arr = []
        for jsonitem in jsondata["instances"]:
            log.info(jsonitem["features"])
            arr.append(jsonitem["features"])
        return arr
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        np_array = decoder.decode(request_body, request_content_type)
        return np_array


def output_fn(prediction, content_type):
    log.info(f"output_fn:\n{prediction}")
    return {"predictions": prediction}
