# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Tuple
import os
import json
from mlmonitor.exceptions import (
    MissingConfigurationError,
    ConfigurationError,
    MissingConfigurationFileError,
)


def get_env() -> str:
    if (env := os.environ.get("ENV", "saas")) not in ["saas", "prem"]:
        raise ConfigurationError(
            f"ENV environment variable set to '{env}'.should be set to 'saas' (IBM Cloud) or 'prem' (On premise cluster)"
        )
    return env


def get_config_file() -> Dict:
    if config_file := os.environ.get("MONITOR_CONFIG_FILE"):
        with open(config_file) as json_file:
            return json.load(json_file)
    else:
        raise MissingConfigurationFileError.missing_configuration_file()


def get_aws_credentials() -> Tuple:
    aws_fields = ("access_key", "secret_key", "region_name", "role")

    aws_envars = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_ROLE",
    )

    if all(k in os.environ for k in aws_envars):
        return (
            os.environ["AWS_ACCESS_KEY_ID"],
            os.environ["AWS_SECRET_ACCESS_KEY"],
            os.environ["AWS_DEFAULT_REGION"],
            os.environ["AWS_ROLE"],
        )

    else:
        credentials = get_config_file()

    if any(k not in credentials.get("aws").keys() for k in aws_fields):
        raise MissingConfigurationError.missing_configuration_value(
            aws_fields,
            aws_envars,
            "aws",
        )
    os.environ["AWS_ACCESS_KEY_ID"] = credentials.get("aws").get("access_key")
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.get("aws").get("secret_key")
    os.environ["AWS_DEFAULT_REGION"] = credentials.get("aws").get("region_name")
    os.environ["AWS_ROLE"] = credentials.get("aws").get("role")

    return (
        credentials.get("aws").get("access_key"),
        credentials.get("aws").get("secret_key"),
        credentials.get("aws").get("region_name"),
        credentials.get("aws").get("role"),
    )


def get_azure_credentials() -> Tuple:
    azure_envars = (
        "AZ_WORKSPACE_NAME",
        "AZ_RESOURCE_GROUP",
        "AZ_SUBSCRIPTION_ID",
        "AZ_TENANT_ID",
        "AZ_SP_ID",
        "AZ_SP_SECRET",
    )
    azure_fields = (
        "workspace_name",
        "resource_group",
        "subscription_id",
        "tenant_id",
        "client_id",
        "client_secret",
    )
    if all(k in os.environ for k in azure_envars):
        return (
            os.environ["AZ_WORKSPACE_NAME"],
            os.environ["AZ_RESOURCE_GROUP"],
            os.environ["AZ_SUBSCRIPTION_ID"],
            os.environ["AZ_TENANT_ID"],
            os.environ["AZ_SP_ID"],
            os.environ["AZ_SP_SECRET"],
        )

    else:
        credentials = get_config_file()

    if any(k not in credentials.get("azure").keys() for k in azure_fields):
        raise MissingConfigurationError.missing_configuration_value(
            azure_fields,
            azure_envars,
            "azure",
        )

    os.environ["AZ_WORKSPACE_NAME"] = credentials.get("azure").get("workspace_name")
    os.environ["AZ_RESOURCE_GROUP"] = credentials.get("azure").get("resource_group")
    os.environ["AZ_SUBSCRIPTION_ID"] = credentials.get("azure").get("subscription_id")
    os.environ["AZ_TENANT_ID"] = credentials.get("azure").get("tenant_id")
    os.environ["AZ_SP_ID"] = credentials.get("azure").get("client_id")
    os.environ["AZ_SP_SECRET"] = credentials.get("azure").get("client_secret")

    return (
        credentials.get("azure").get("workspace_name"),
        credentials.get("azure").get("resource_group"),
        credentials.get("azure").get("subscription_id"),
        credentials.get("azure").get("tenant_id"),
        credentials.get("azure").get("client_id"),
        credentials.get("azure").get("client_secret"),
    )


def get_connection_details(
    env: str = "saas", iam_url: str = "https://iam.bluemix.net/oidc/token"
) -> Tuple:
    assert env in {"saas", "prem"}
    if env == "saas":
        os.environ["AUTH_ENDPOINT"] = iam_url  # no need ot set CP4D endpoint
        mandatory_envar = ("API_KEY",)
    elif env == "prem":
        mandatory_envar = ("API_KEY", "AUTH_ENDPOINT")

    if all(k in os.environ for k in mandatory_envar):
        return (
            os.environ["API_KEY"],
            os.getenv("AUTH_ENDPOINT"),
        )

    credentials = get_config_file()
    mandatory_fields = ("apikey", "ibm_auth_endpoint") if env == "prem" else ("apikey",)
    if all(k in credentials.get(env).keys() for k in mandatory_fields):
        return (
            credentials.get(env).get("apikey"),
            credentials.get(env).get("ibm_auth_endpoint", iam_url),
        )
    else:
        raise MissingConfigurationError.missing_configuration_value(
            mandatory_fields,
            mandatory_envar,
            env,
        )


def get_cos_details(env: str = "saas") -> Tuple:
    # for CP4 on prem API_KEY is different from IBM CLOUD API KEY required for COS
    # It must be explicitly set in CLOUD_API_KEY env variable
    mandatory_envar = (
        ("COS_RESOURCE_CRN", "COS_ENDPOINT", "BUCKET_NAME", "CLOUD_API_KEY")
        # for CP4D SaaS API_KEY is the same as IBM CLOUD API KEY for COS
        if env == "prem"
        else ("COS_RESOURCE_CRN", "COS_ENDPOINT", "BUCKET_NAME", "API_KEY")
    )

    if all(k in os.environ for k in mandatory_envar):
        saas_api_key = (
            os.getenv("API_KEY") if env == "saas" else os.getenv("CLOUD_API_KEY")
        )
        return (
            os.environ["COS_RESOURCE_CRN"],
            os.environ["COS_ENDPOINT"],
            os.environ["BUCKET_NAME"],
            saas_api_key,
        )

    else:
        credentials = get_config_file()

    if all(
        k in credentials.get("saas").keys()
        for k in ("cos_resource_crn", "cos_endpoint", "bucket_name", "apikey")
    ):
        return (
            credentials.get("saas").get("cos_resource_crn"),
            credentials.get("saas").get("cos_endpoint"),
            credentials.get("saas").get("bucket_name"),
            credentials.get("saas").get("apikey"),
        )
    else:
        raise MissingConfigurationError.missing_configuration_value(
            ("cos_resource_crn", "cos_endpoint", "bucket_name", "apikey"),
            mandatory_envar,
            "saas",
        )


def get_db2_details():
    credentials = get_config_file()
    return credentials.get("db2")


def get_cp4d_on_prem_details(env: str = "prem") -> Tuple:
    if env == "saas":
        return None, None
    if all(k in os.environ for k in ("CP4D_VERSION", "CP4D_USERNAME")):
        return os.environ["CP4D_VERSION"], os.environ["CP4D_USERNAME"]

    else:
        credentials = get_config_file()

    if all(k in credentials.get("prem").keys() for k in ("version", "username")):
        return credentials.get("prem").get("version"), credentials.get("prem").get(
            "username"
        )
    else:
        raise MissingConfigurationError.missing_configuration_value(
            ("version", "username"),
            ("CP4D_VERSION", "CP4D_USERNAME"),
            env,
        )


def get_wkc_details(env: str = "saas") -> Tuple:
    assert env in {"saas", "prem"}
    if all(k in os.environ for k in ("MODEL_ENTRY_ID", "CATALOG_ID")):
        return os.environ["MODEL_ENTRY_ID"], os.environ["CATALOG_ID"]

    else:
        credentials = get_config_file()

    return (
        credentials.get(env).get("model_entry_id"),
        credentials.get(env).get("catalog_id"),
    )


def get_wml_details(env: str = "saas") -> Tuple:
    assert env in {"saas", "prem"}
    default_wml_instance_id = "openshift" if env == "prem" else None

    mandatory_envar = (
        ("AUTH_ENDPOINT", "WML_SPACE_ID")
        # for CP4D prem AUTH_ENDPOINT is the CP4D url used for all services
        if env == "prem"
        else ("WML_SPACE_ID", "WML_URL")
    )

    if all(k in os.environ for k in mandatory_envar):
        wml_url = (
            os.getenv("WML_URL", os.environ["AUTH_ENDPOINT"])
            if env == "prem"
            else os.environ["WML_URL"]
        )
        return (
            os.getenv("WML_INSTANCE_ID", default_wml_instance_id),
            wml_url,
            os.getenv("WML_SPACE_ID"),
        )

    credentials = get_config_file()

    mandatory_fields = (
        ("wml_url", "default_space") if env == "saas" else ("default_space",)
    )

    if any(k not in credentials.get(env).keys() for k in mandatory_fields):
        raise MissingConfigurationError.missing_configuration_value(
            mandatory_fields,
            mandatory_envar,
            env,
        )
    default_wml_url = (
        credentials.get("prem").get("ibm_auth_endpoint") if env == "prem" else None
    )
    return (
        credentials.get(env).get("wml_instance_id", default_wml_instance_id),
        credentials.get(env).get("wml_url", default_wml_url),
        credentials.get(env).get("default_space"),
    )


def get_wos_details(env: str = "saas") -> Tuple:
    assert env in {"saas", "prem"}
    mandatory_envar = (
        ("AUTH_ENDPOINT",)
        # for CP4D prem AUTH_ENDPOINT is the CP4D url used for all services
        if env == "prem"
        else ("WOS_INSTANCE_ID",)
    )

    if all(k in os.environ for k in mandatory_envar):
        wos_url = (
            os.getenv("WOS_URL", os.environ["AUTH_ENDPOINT"])
            if env == "prem"
            else "https://api.aiopenscale.cloud.ibm.com"
        )
        return (
            wos_url,
            os.getenv("WOS_INSTANCE_ID", "00000000-0000-0000-0000-000000000000"),
        )

    credentials = get_config_file()

    mandatory_fields = (
        ("ibm_auth_endpoint",)
        # for CP4D prem AUTH_ENDPOINT is the CP4D url used for all services
        if env == "prem"
        else ("wos_instance_id",)
    )

    if any(k not in credentials.get(env).keys() for k in mandatory_fields):
        raise MissingConfigurationError.missing_configuration_value(
            mandatory_fields, mandatory_envar, env
        )

    default_wos_instance_id = (
        "00000000-0000-0000-0000-000000000000" if env == "prem" else None
    )
    default_wos_url = (
        "https://api.aiopenscale.cloud.ibm.com"
        if env == "saas"
        else credentials.get(env).get("ibm_auth_endpoint")
    )

    return (
        credentials.get(env).get("wos_url", default_wos_url),
        credentials.get(env).get("wos_instance_id", default_wos_instance_id),
    )


def assign_verify_ssl(default_value: bool = True) -> bool:
    verify = os.getenv("VERIFY_CP4D_SSL")
    if verify and verify == "True":
        verify = True
    elif verify and verify == "False":
        verify = False
    else:
        verify = default_value
    return verify
