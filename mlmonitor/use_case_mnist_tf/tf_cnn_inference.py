# SPDX-License-Identifier: Apache-2.0
import json
from collections import namedtuple
import numpy as np
import requests
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

Context = namedtuple(
    "Context",
    "model_name, model_version, method, rest_uri, grpc_uri, "
    "custom_attributes, request_content_type, accept_header",
)


def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, context)


def _process_input(data, context):
    decoded_data = data.read().decode("utf-8")
    log.info(
        f"_process_input request_content_type {context.request_content_type} decoded_data:\n{decoded_data} "
    )

    #######################################################################################
    # Handle request format from Watson Openscale feedback Data (application/json)        #
    #######################################################################################
    # data = [json.loads(f"[{x.split(';')[0]}]") for x in decoded_data.split('\n')]

    if context.request_content_type == "application/json":
        try:
            data = json.loads(decoded_data).get("input_data")[0].get("values")

            return json.dumps({"instances": data})
        except Exception as e:
            raise ValueError(
                f'Exception _process_input json "{e}"'
                f"Input format {type(decoded_data)}"
                f"{decoded_data}"
            ) from e

    elif context.request_content_type == "text/csv":

        try:
            data = [json.loads(f"{x.split(';')[0]}") for x in decoded_data.split("\n")]
            return json.dumps({"instances": data})
        except Exception as e:
            raise ValueError(
                f'Exception _process_input csv "{e}"'
                f"Input format {type(decoded_data)}"
                f"{decoded_data}"
            ) from e

    raise ValueError(
        '{{"error": "unsupported content type {}"}}'.format(
            context.request_content_type or "unknown"
        )
    )


def _process_output(data, context):
    if data.status_code != 200:
        raise ValueError(
            f"ValueError in _process_output : {data.content.decode('utf-8')}"
        )

    d = data.content.decode("utf-8")
    prediction = json.loads(d).get("predictions")
    classes = np.argmax(np.array(prediction), axis=1).tolist()
    values = list(zip([int(x) for x in classes], prediction))
    fields = ["prediction", "probability"]
    output = {"fields": fields, "values": values}
    response = {"predictions": [output]}
    #####################################################################
    # Handle response format required by Watson Openscale Evaluate      #
    #####################################################################
    # values = list(zip([int(x) for x in classes], prediction))
    # fields = ['_original_prediction', '_original_probability']
    # fields = ['prediction', 'probability']
    # output = {'fields': fields, 'values': values }
    # response = {'predictions': [output]}

    if context.request_content_type == "application/json":
        response_content_type = "application/json"
        log.info(
            f"_process_output request_content_type 'application/json' response:\n{json.dumps(response)} "
        )
        return json.dumps(response), response_content_type

    elif context.request_content_type == "text/csv":
        response_content_type = "application/json"
        log.info(
            f"_process_output request_content_type 'text/csv' response:\n{json.dumps(response)} "
        )
        return json.dumps(response), response_content_type

    raise ValueError(
        '{{"error": "unsupported content type {}"}}'.format(
            context.request_content_type or "unknown"
        )
    )
