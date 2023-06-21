# SPDX-License-Identifier: Apache-2.0
import ibm_boto3
from botocore.client import Config
from mlmonitor.src import IAM_URL, ENV
from mlmonitor.config import get_cos_details

COS_RESOURCE_CRN, COS_ENDPOINT, BUCKET_NAME, CLOUD_API_KEY = get_cos_details(env=ENV)

cos_client = ibm_boto3.client(
    service_name="s3",
    ibm_api_key_id=CLOUD_API_KEY,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=IAM_URL,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT,
)

cos_resource = ibm_boto3.resource(
    "s3",
    ibm_api_key_id=CLOUD_API_KEY,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=IAM_URL,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT,
)
