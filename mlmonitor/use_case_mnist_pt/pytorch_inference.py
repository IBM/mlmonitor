# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import torch
import numpy as np

from pt_models import ConvNet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    log.info(
        f"input_fn request_content_type {request_content_type} request_body:\n{request_body}"
    )
    data = json.loads(request_body).get("input_data")[0].get("values")

    data = torch.tensor(data, dtype=torch.float32, device=device)
    if data.shape[3] == 1:  # 1 channel at the end of tensor
        log.info(f"input_fn {data.shape} channel last permuting to channel first")
        data = data.permute(0, 3, 1, 2)
    else:
        log.info(
            f"input_fn {data.shape} channel first sent (already in expected format)"
        )

    log.info(f"input_fn data shape sent to model {data.shape}")

    return data


# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        log.info(f"predict_fn data shape sent to model {input_object.shape}")
        prediction = model(input_object)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"

    probabilities = torch.exp(predictions).cpu().numpy().tolist()

    log_probas = predictions.cpu().numpy().tolist()

    classes = np.argmax(np.array(log_probas), axis=1).tolist()
    fields = ["prediction", "probability"]
    values = list(zip([int(x) for x in classes], probabilities))
    output = {"fields": fields, "values": values}
    return json.dumps({"predictions": [output]})


# defining model and loading weights to it.
def model_fn(model_dir):
    model = ConvNet()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    return model
