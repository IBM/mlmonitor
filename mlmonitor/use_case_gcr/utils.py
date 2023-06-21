# SPDX-License-Identifier: Apache-2.0
import base64
from botocore.exceptions import ClientError
import boto3
import json
import os


def _get_secret(
    secret_name: str,
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    region_name: str = "ca-central-1",
):
    """
    helper function that retrieves the secret value from AWS Secrets Manager.
    It takes in a string representing the name of the secret, and returns either a string or binary representation of
    the secret value.

    :param secret_name:str: Store the name of the secret in aws secrets manager
    :param aws_access_key_id:str=None: Pass the aws access key id to the function
    :param aws_secret_access_key:str=None: Pass the secret key to _get_secret function
    :param region_name:str=&quot;ca-central-1&quot;: Specify the aws region of the secret
    :param : Get the secret from aws secrets manager
    :return: decode secret value
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "AccessDeniedException":
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        else:
            print(e.response)
    else:
        if "SecretString" in get_secret_value_response:
            return get_secret_value_response["SecretString"]
        decoded_binary_secret = base64.b64decode(
            get_secret_value_response["SecretBinary"]
        )
        return decoded_binary_secret


def read_columns():
    filepath = (
        "/opt/ml/code/model_signature.json"
        if os.path.exists("/opt/ml/code/model_signature.json")
        else os.path.join(os.path.dirname(__file__), "model_signature.json")
    )
    with open(filepath) as json_file:
        signature = json.load(json_file)

    return signature.get("signature").get("feature_columns")
