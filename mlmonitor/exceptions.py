# SPDX-License-Identifier: Apache-2.0
import json
from typing import List, Tuple
from os.path import abspath, dirname, join

INTERNAL_ERROR = 1
INVALID_SIGNATURE_VALUE = 1000
MISSING_CONFIG_VALUE = 2000
MISSING_CONFIG_FILE = 3000
MISMATCH_AWS_KEY_MANAGER = 4000


class MLMonitorException(Exception):
    """
    Base mlmonitor error.
    """

    def __init__(self, message, error_code=INVALID_SIGNATURE_VALUE, **kwargs):
        """
        :param message: message or exception describing the error that occurred included in the exception's serialized JSON.
        :param error_code: An appropriate error code included in the exception's serialized JSON representation.
        :param kwargs: Additional key-value pairs to include in the serialized JSON representation of the MLMonitorException.
        """
        self.error_code = error_code
        self.message = str(message)
        self.json_kwargs = kwargs
        super().__init__(str(message))

    def serialize_as_json(self):
        exception_dict = {"error_code": self.error_code, "message": self.message}
        exception_dict |= self.json_kwargs
        return json.dumps(exception_dict)

    @classmethod
    def invalid_signature_value(
        cls,
        field_name: str,
        passed_value: str,
        supported_values: List,
        section_name: str,
        **kwargs,
    ):
        """
        Constructs an `MLMonitorException` object with the `MISSING_CONFIG_VALUE` error code.

        :param field_name:str of the field in signature file
        :param passed_value:str value that was passed in signature file
        :param supported_values:list expected values
        :param section_name:str section name in signature file
        :return:ModelSignatureError A ModelSignatureError exception object
        """
        message = f"signature field `{field_name}` in `{section_name}` section expected in {supported_values} but got `{passed_value}`"
        return cls(message, error_code=MISSING_CONFIG_VALUE, **kwargs)

    @classmethod
    def missing_configuration_value(
        cls, field_names: Tuple, env_vars: Tuple, section_name: str
    ):
        """
        Constructs an `MissingConfigurationError` object with the `MISSING_CONFIG_VALUE` error code.

        :param field_names:Tuple of the field in signature file
        :param env_vars:str Tuple that was passed in signature file
        :param section_name:str section name in credentials.cfg file
        :return:MissingConfigurationError A MissingConfigurationError exception object
        """
        message = (
            f"\n\tCould not find the necessary information to connect in `{section_name}` section.\n\t"
            f"Please enter your {field_names} "
            "in a configuration file credentials.cfg or\n\t"
            f"in {env_vars} environment variables."
        )

        return cls(message, error_code=MISSING_CONFIG_VALUE)

    @classmethod
    def mismatch_keymanager_secret_value(cls, secret_name: str, secret_value: str):
        """
        Constructs an `MissingKeyManagerSecretName` object with the `MISSING_AWS_KEY_MANAGER` error code.

        :param secret_name: AWS Secrets Key Manager secret name
        :param secret_value: AWS Secrets Key Manager secret value
        :return:MisMatchKeyManagerSecretValue A MisMatchKeyManagerSecretValue exception object
        """
        message = (
            f"\nsecret [{secret_name}] already set in AWS Secrets manager with a different value than specified API - cannot be automatically replaced by {secret_value[:4]}XXXXXXXXXX\n"
            f"if you want to replace [{secret_name}] run `replace_secrets_manager()` method of SageMakerModelUseCase or update API Key from AWS console\n"
            "- https://github.com/awsdocs/aws-doc-sdk-examples/tree/main/python/example_code/secretsmanager#code-examples\n"
            "- https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/secretsmanager.html\n"
        )

        return cls(message, error_code=MISMATCH_AWS_KEY_MANAGER)

    @classmethod
    def missing_configuration_file(cls):
        """
        Constructs an `MissingConfigurationFileError` object with the `MISSING_CONFIG_FILE` error code.

        :return:MissingConfigurationFileError A MissingConfigurationFileError exception object
        """

        with open(
            join(abspath(dirname(__file__)), "credentials_example.cfg")
        ) as json_file:
            example_cfg = json.load(json_file)

        message = (
            f"you should include MONITOR_CONFIG_FILE in environment variable with configuration file path\n"
            "config file should follow this structure:\n"
            f"{json.dumps(example_cfg, indent=3)}"
        )

        return cls(message, error_code=MISSING_CONFIG_FILE)


class UnauthorizedError(MLMonitorException):
    """Action was not authorized (did you provide correct credentials?)."""


class ConfigurationError(MLMonitorException):
    """Configuration is invalid or not found."""


class MissingConfigurationError(ConfigurationError):
    """Configuration field is missing."""


class MissingConfigurationFileError(ConfigurationError):
    """Configuration file is missing."""


class MisMatchKeyManagerSecretValue(ConfigurationError):
    """AWS KeyManager conflict impossible to store API Key."""


class ModelSignatureError(ConfigurationError):
    """Error in model signature definition."""
