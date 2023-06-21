# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os

import torch

from use_case_mnist_ptlt.pt_models import PytorchLightning_CNN_MNIST

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# defining model and loading weights to it.
def model_fn(model_dir):
    checkpoint = torch.load(os.path.join(model_dir, "model_checkpoint.ckpt"))
    hparams = checkpoint["hyper_parameters"]
    ptlt_model = PytorchLightning_CNN_MNIST(arguments=hparams.get("arguments"))
    model = ptlt_model.model
    model.to(device).eval()
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
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
        prediction = model(input_object)
        # prediction = model.forward(input_object)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)
