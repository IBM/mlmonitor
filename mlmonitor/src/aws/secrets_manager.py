# SPDX-License-Identifier: Apache-2.0
import json
from botocore.exceptions import ClientError
from logging import Logger
import boto3
from typing import Tuple


def sm_secret_name_exists(
    client: boto3.client, secret_name: str, logger: Logger
) -> bool:
    """
    returns whether secret_name exists in AWS Secrets Manager

    :param client:Session: A Boto3 Secrets Manager client.
    :param secret_name: str: secret name to create in Secrets Manager
    :param logger : Logger: Log the error message if there is an exception
    :return: bool
    """
    try:
        client.get_secret_value(SecretId=secret_name)
    except ClientError:
        logger.info(f"secret [{secret_name}] does not exist in Secrets Manager")
        return False
    return True


def sm_create(
    client: boto3.client,
    secret_name: str,
    secret_key_name: str,
    secret_key_value: str,
    logger: Logger,
):
    """
    creates a secret in AWS Secrets Manager with secret name - adds key,value pair specified in argument

    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/secretsmanager.html

    :param client:Session: A Boto3 Secrets Manager client.
    :param secret_name: str: secret name to create in Secrets Manager
    :param secret_key_name: str: key name for the secret in Secrets Manager
    :param secret_key_value: str: key value for the secret in Secrets Manager
    :param logger : Logger: Log the error message if there is an exception
    :return: None
    """
    secret_string = json.dumps({secret_key_name: secret_key_value})
    client.create_secret(Name=secret_name, SecretString=secret_string)
    logger.info(f"Created secret [{secret_name}] - secret key [{secret_key_name}]")


def sm_secret_key_name_exists(
    client: boto3.client,
    secret_name: str,
    secret_key_name: str,
    secret_value: str,
    logger: Logger,
) -> Tuple[bool, bool]:
    """
    returns whether secret_key_name already exists in secret_name of AWS Secrets Manager

    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/secretsmanager.html

    :param client:Session: A Boto3 Secrets Manager client.
    :param secret_name: str: secret name to create in Secrets Manager
    :param secret_key_name: str: key name for the secret in Secrets Manager
    :param secret_value: str: secret value to be set for secret_key_name
    :param logger : Logger: Log the error message if there is an exception
    :return: bool,bool
    """
    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        logger.error(f"secret name `{secret_name}` secret : {e}")
        return False, False

    if "SecretString" in response:
        secret_strings = json.loads(response.get("SecretString"))
        current_value = secret_strings.get(secret_key_name)
        return secret_key_name in secret_strings, current_value == secret_value

    return False, False


def sm_update(
    client: boto3.client,
    secret_name: str,
    secret_key_name: str,
    secret_key_value: str,
    logger: Logger,
):
    """
    creates a secret in AWS Secrets Manager with secret name - adds key,value pair specified in argument

    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/secretsmanager.html

    :param client:Session: A Boto3 Secrets Manager client.
    :param secret_name: str: secret name to create in Secrets Manager
    :param secret_key_name: str: key name for the secret in Secrets Manager
    :param secret_key_value: str: key value for the secret in Secrets Manager
    :param logger : Logger: Log the error message if there is an exception
    :return: None
    """

    response = client.get_secret_value(SecretId=secret_name)

    if "SecretString" in response:
        secret_strings = json.loads(response.get("SecretString"))
        if secret_key_name in secret_strings:
            logger.warning(
                f"Updating secret [{secret_name}] - secret key [{secret_key_name}]"
            )
        else:
            logger.warning(
                f"Adding secret [{secret_name}] - secret key [{secret_key_name}]"
            )

        secret_strings[secret_key_name] = secret_key_value
        client.put_secret_value(
            SecretId=secret_name, SecretString=json.dumps(secret_strings)
        )


def sm_delete(
    client: boto3.client,
    secret_name: str,
    logger: Logger,
    without_recovery: bool = False,
):
    """
     deletes a secret in AWS Secrets Manager with secret name

     https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/secretsmanager.html

     :param client:Session: A Boto3 Secrets Manager client.
     :param secret_name: str: secret name to create in Secrets Manager
     :param logger : Logger: Log the error message if there is an exception
    :param without_recovery: Permanently deletes the secret immediately when True;
                              otherwise, the deleted secret can be restored within
                              the recovery window. The default recovery window is
                              30 days.
     :return: None
    """
    try:
        client.delete_secret(
            SecretId=secret_name, ForceDeleteWithoutRecovery=without_recovery
        )
        logger.info(f"Deleted secret {secret_name} without_recovery {without_recovery}")
    except ClientError as e:
        logger.error(f"error deleting secret name `{secret_name}` secret : {e}")
