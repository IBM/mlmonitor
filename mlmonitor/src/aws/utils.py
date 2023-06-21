# SPDX-License-Identifier: Apache-2.0
import re
import os
import sagemaker
from sagemaker.inputs import TrainingInput
from boto3.session import Session as boto_Session

from mlmonitor.src import logger
from mlmonitor.src.utils.validation import is_csv, is_directory


def s3_upload_training_job_datasets(
    data_path: str,
    sagemaker_session: sagemaker.Session,
    boto_session: boto_Session,
    bucket: str,
    prefix: str,
    training_data_location: str,
    val_data_location: str = None,
    test_data_location: str = None,
) -> dict:
    """

        - Uploads the training, validation and test datasets to S3.
        - Returns a dictionary of the uploaded datasets.
        The key is the dataset type (train, validation or test) and value is an s3_input object for that dataset.

    :param data_path:str: Specify the directory where the training, validation and test data is located
    :param sagemaker_session:sm_Session: sagemaker session
    :param boto_session:boto_Session: Access the aws resources
    :param bucket:str: s3 bucket to which the data is uploaded
    :param prefix:str: prefix directory in the s3 bucket
    :param training_data_location:str: location of the training data to upload
    :param val_data_location:str=None: location of the validation data to upload
    :param test_data_location:str=None: location of the test data to upload
    :return: dictionary to be used to trigger Sagemaker training job
    """
    s3 = boto_session.resource("s3")
    job_dataset_dict = {}
    # training for Sagemaker
    datasets = {
        "train": training_data_location,
        "validation": val_data_location,
        "test": test_data_location,
    }
    uploaded = {}
    for dataset_type, dataset_location in datasets.items():

        if dataset_location:

            if dataset_location in uploaded:
                logger.info(
                    f"skipping uploading {dataset_type} similar to {uploaded[dataset_location]} "
                )
                job_dataset_dict[dataset_type] = job_dataset_dict.get(
                    uploaded[dataset_location]
                )
            else:
                full_path = (
                    data_path
                    # check if data_dir (last part of data_path) equals dataset_location dir
                    if data_path.split("/")[-1] == dataset_location
                    else os.path.join(data_path, dataset_location)
                )
                dataset_is_csv = is_csv(full_path)
                dataset_is_dir = is_directory(full_path)

                if dataset_is_csv:

                    s3_input_train = TrainingInput(
                        s3_data=f"s3://{bucket}/{prefix}/{dataset_type}",
                        content_type="csv",
                    )
                    s3.Bucket(bucket).Object(
                        os.path.join(prefix, dataset_type, dataset_location)
                    ).upload_file(full_path)
                elif dataset_is_dir:
                    s3_input_train = sagemaker_session.upload_data(
                        path=full_path, bucket=bucket, key_prefix=prefix
                    )
                else:
                    message = f"{dataset_type} dataset {dataset_location} is csv file:[{dataset_is_csv}] is dir:[{dataset_is_dir}]"
                    logger.error(message)
                    raise TypeError(message)

                uploaded[dataset_location] = dataset_type
                job_dataset_dict[dataset_type] = s3_input_train
                logger.info(
                    f"uploaded {dataset_type} dataset from {data_path}/{dataset_location} to s3://{bucket}/{prefix}/{dataset_type}/{dataset_location}"
                )
        else:
            logger.info(f"skipping uploading {dataset_type} dataset from None")

    return job_dataset_dict


def jobname_from_modeldata(model_data: str) -> str:
    """
    extracts the job name from a model data path.

    :param model_data:str: model location path in s3 generated from Sagemaker job
    :return: jobname extracted from the model data
    """
    if match := re.match(
        r"^s3://.*/([A-Za-z_0-9.-]+-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3})/.*model.tar.gz$",
        model_data,
    ):
        return match[1]
    else:
        raise ValueError("Not able to extract Endpoint Name from provided model output")
