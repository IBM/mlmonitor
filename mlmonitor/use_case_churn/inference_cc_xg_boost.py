# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np

# import sagemaker_xgboost_container.encoder as xgb_encoders
# from sagemaker_inference import content_types, decoder
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model_file = "xgboost-model-bst"
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, model_file))
    return model


def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.
    Return a DMatrix (an object that can be passed to predict_fn).
    """
    log.info(f"Content type {request_content_type} received.")
    if request_content_type == "text/libsvm":
        # return xgb_encoders.libsvm_to_dmatrix(request_body)
        return ValueError(f"Content type {request_content_type} is not supported.")
    if request_content_type != "text/csv":
        raise ValueError(f"Content type {request_content_type} is not supported.")
    inputs = np.array(
        [[float(y) for y in row.split(",")] for row in request_body.split("\n")]
    )
    return xgb.DMatrix(inputs)


def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.
    Return a two-dimensional NumPy array where the first columns are predictions
    and the remaining columns are the feature contributions (SHAP values) for that prediction.
    """
    prediction = model.predict(input_data, pred_contribs=False, validate_features=False)
    return prediction


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    if content_type == "text/csv":
        return ",".join(str(x) for x in predictions)
    else:
        raise ValueError(f"Content type {content_type} is not supported.")
