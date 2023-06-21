# SPDX-License-Identifier: Apache-2.0
from ibm_botocore.client import ClientError
import os

from mlmonitor.data import cos_client, cos_resource
from mlmonitor.data import BUCKET_NAME


def get_item(item_name: str, bucket_name: str = BUCKET_NAME):
    print("Retrieving item from bucket: {0}, key: {1}".format(bucket_name, item_name))
    try:
        obj = cos_client.get_object(Bucket=bucket_name, Key=item_name)["Body"]
        return obj
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve file contents: {0}".format(e))


def put_item(item_name: str, item_path: str, bucket_name: str = BUCKET_NAME):
    print("Uploading item to bucket: {0}, key: {1}".format(bucket_name, item_name))
    try:
        with open(os.path.join(item_path, item_name), "rb") as file_data:
            cos_resource.Object(bucket_name, item_name).upload_fileobj(
                Fileobj=file_data
            )
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to upload file contents: {0}".format(e))


def write_item_resource(
    item_name: str, bucket_name: str = BUCKET_NAME, dest_path: str = "/tmp"
):
    if not os.path.isdir(dest_path):
        raise ValueError(f"invalid destination path {dest_path}")
    # ibm_boto3.resource
    obj = cos_resource.Object(bucket_name, item_name).get()
    file = obj["Body"].read()

    with open(item_name, "w+b") as f:
        f.write(file)
