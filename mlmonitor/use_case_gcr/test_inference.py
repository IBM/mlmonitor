# SPDX-License-Identifier: Apache-2.0
import os
import pandas as pd
import json
from os.path import dirname

from use_case_gcr.inference_aws_gcr import (
    model_fn,
    input_fn,
    predict_fn,
    output_fn,
)
from use_case_gcr.utils import read_columns

if __name__ == "__main__":

    COLUMNS = read_columns()
    PROJECT_ROOT = dirname(dirname(__file__))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets")
    MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")

    scoring_df = pd.read_csv(
        os.path.join(DATA_ROOT, "gcr_explicit_payload_logging.csv")
    )

    scoring_payload = {
        "instances": [{"features": x} for x in scoring_df[COLUMNS].values.tolist()]
    }
    model = model_fn(f"{MODEL_ROOT}/model_gcr")
    print(scoring_payload)
    input_payload = input_fn(json.dumps(scoring_payload), "application/json")
    outputs = predict_fn(input_payload, model)
    preds = output_fn(outputs, "application/json")
    print(preds)
