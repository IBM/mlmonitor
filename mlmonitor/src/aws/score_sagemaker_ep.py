# SPDX-License-Identifier: Apache-2.0
from mlmonitor.src.aws.scoring import get_wos_response, _score_unstructured
from mlmonitor.src.model.config import ModelConfig


def score_sagemaker_endpoint(
    model_config: ModelConfig,
    deployment_name: str,
    aws_credentials: dict,
    inference_samples: int = 2,
) -> dict:

    """
    Sends a scoring request with <inference_samples samples> to the deployed Sagemaker Endpoint with name <deployment_name>

    :param model_config: ModelConfig: model configuration
    :param deployment_name: str: EndpointName of AWS online inference endpoint.
    :param aws_credentials: dict: AWS credentials to use to invoke deployed online inference endpoint in AWS Sagemaker. this dictionary should be formatted as follow:
        {"aws_access_key_id": <your access_key>,
        "aws_secret_access_key": <your secret>,
        "region_name": <your region name>}
    :param inference_samples: int: Number of inference samples to send in the scoring request to the deployed Endpoint (deployment_name)
    :return: dictionary containing the predicted values formatted for Watson OpenScale
    scoring response received and transformed to Watson OpenScale accepted format for feedback logging
        {'fields': ['_original_prediction', '_original_probability'], 'values': [[1, 0.984], [1, 0.997]]}
    """
    test_data = model_config._get_data(
        dataset_type="test", num_samples=inference_samples
    )

    if model_config.data_type == "structured":

        df = test_data.loc[:, model_config.feature_columns]

        return get_wos_response(
            df=df,
            aws_access_key_id=aws_credentials.get("aws_access_key_id"),
            aws_secret_access_key=aws_credentials.get("aws_secret_access_key"),
            region_name=aws_credentials.get("region_name"),
            endpoint_name=deployment_name,
            prediction_field=model_config.prediction_field,
            probability_field=model_config.probability_fields[0],
        )

    elif model_config.data_type == "unstructured_image":

        samples, labels = test_data
        result = _score_unstructured(
            payload=samples, endpoint_name=deployment_name, **aws_credentials
        )

        return {
            "fields": result.get("predictions")[0].get("fields"),
            "values": result.get("predictions")[0].get("values"),
        }

    else:
        raise ValueError(
            "supported data_type are structured or unstructured_image (must be passed in model signature)"
        )
