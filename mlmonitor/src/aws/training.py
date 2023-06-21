# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Tuple, Optional
import sagemaker
from sagemaker import image_uris
from boto3.session import Session as boto_Session


def generate_sk_training_params(
    framework: str,
    framework_version: str,
    estimator_params: Dict,
    train_dict: Dict,
    sagemaker_session: sagemaker.Session,
) -> Tuple[Dict, Dict]:
    """
    takes in the framework, framework_version, estimator_params and train_dict
    returns a tuple of 2 dictionaries.

    1) contains - image URI for the container that used for training and inference
                - parameters required by Scikit Learn Estimator (sagemaker.estimator.Estimator used in mlmonitor).
                - Sagemaker session

            {
                leveraging Estimators : https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
                NOT sagemaker.sklearn.estimator.SKLearn
                Scikit-learn specific doc : https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html

                "image_uri": retrieved image uri based on framework and framework version
                "entry_point":  Path (absolute or relative) to the Python source file which should be executed as the entry point to training.
                "source_dir": Path (absolute, relative or an S3 URI) to a directory with any other training source code dependencies aside from the entry point file
                "hyperparameters": Hyperparameters that will be used for training

                EstimatorBase doc : https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html

                "role": EstimatorBase parameter,
                "base_job_name": EstimatorBase parameter,
                "instance_count": EstimatorBase parameter,
                "instance_type": EstimatorBase parameter,
                "output_path": EstimatorBase parameter
            }

    2) contains data location 'train', 'validation', 'test'

    :param framework: str: container image types to use for training
    :param framework_version: str: version of the scikit-learn framework used
    :param estimator_params: Dict: estimator parameters
    :param train_dict: Dict: training parameters - might vary from one framework to another
    :param sagemaker_session: sagemaker.Session: instantiated sagemaker
    :return: A tuple of two dictionaries (estimator_params_sk,train_dict_sk)
    """
    estimator_params_sk = estimator_params.copy()
    train_dict_sk = train_dict.copy()

    container = image_uris.retrieve(
        framework,
        sagemaker_session.boto_region_name,
        framework_version,
    )
    estimator_params_sk["image_uri"] = container
    estimator_params_sk["sagemaker_session"] = sagemaker_session

    return estimator_params_sk, train_dict_sk


def generate_tf_training_params(
    py_version: str, framework_version: str, estimator_params: Dict, train_dict: Dict
) -> Tuple[Dict, Dict]:
    """
    takes in the framework, framework_version, estimator_params and train_dict
    returns a tuple of 2 dictionaries.

    1) contains - py_version and framework_version used to retrieve image uri
                - all the parameters required by Tensorflow Estimator.

            {
                Tensorflow specific doc : https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/sagemaker.tensorflow.html

                "entry_point":  Path (absolute or relative) to the Python source file which should be executed as the entry point to training.
                "source_dir": Path (absolute, relative or an S3 URI) to a directory with any other training source code dependencies aside from the entry point file
                "py_version":  Python version you want to use for executing your model training code
                "framework_version":  TensorFlow version you want to use for executing your model training code
                "hyperparameters": Hyperparameters that will be used for training

                EstimatorBase doc : https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html

                "base_job_name": EstimatorBase parameter,
                "role": EstimatorBase parameter,
                "instance_count": EstimatorBase parameter,
                "instance_type": EstimatorBase parameter,
                "output_path": EstimatorBase parameter
            }

    2) contains data location 'training', 'validation', 'testing' (Inconsistent with Scikit-Learn train,test)

    :param py_version: str: python version of training job container
    :param framework_version: str: version of the scikit-learn framework used
    :param estimator_params: Dict: estimator parameters
    :param train_dict: Dict: training parameters - might vary from one framework to another
    :return: A tuple of two dictionaries (estimator_params_sk,train_dict_sk)
    """
    estimator_params_tf = estimator_params.copy()

    estimator_params_tf["py_version"] = py_version
    estimator_params_tf["framework_version"] = framework_version
    mapping = {"train": "training", "test": "testing", "validation": "validation"}
    train_dict_tf = {
        mapping.get(key): value for key, value in train_dict.items() if value
    }

    return estimator_params_tf, train_dict_tf


def generate_pt_training_params(
    py_version: str, framework_version: str, estimator_params: Dict, train_dict: Dict
) -> Tuple[Dict, Dict]:
    """
    takes in the framework, framework_version, estimator_params and train_dict
    returns a tuple of 2 dictionaries.

    1) contains - py_version and framework_version used to retrieve PyTorch image uri
                - all the parameters required by PyTorch Estimator.

            {
                PyTorch specific doc : https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html

                "entry_point":  Path (absolute or relative) to the Python source file which should be executed as the entry point to training.
                "source_dir": Path (absolute, relative or an S3 URI) to a directory with any other training source code dependencies aside from the entry point file
                "py_version":  Python version you want to use for executing your model training code
                "framework_version":  PyTorch version you want to use for executing your model training code
                "hyperparameters": Hyperparameters that will be used for training

                EstimatorBase doc : https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html

                "base_job_name": EstimatorBase parameter,
                "role": EstimatorBase parameter,
                "instance_count": EstimatorBase parameter,
                "instance_type": EstimatorBase parameter,
                "output_path": EstimatorBase parameter
            }

    2) contains data location 'training', 'validation', 'testing' (Inconsistent with Scikit-Learn train,test)

    :param py_version: str: python version of training job container
    :param framework_version: str: version of the scikit-learn framework used
    :param estimator_params: Dict: estimator parameters
    :param train_dict: Dict: training parameters - might vary from one framework to another
    :return: A tuple of two dictionaries (estimator_params_sk,train_dict_sk)
    """

    estimator_params_pt = estimator_params.copy()

    estimator_params_pt["py_version"] = py_version
    estimator_params_pt["framework_version"] = framework_version
    mapping = {"train": "training", "test": "testing", "validation": "validation"}
    train_dict_pt = {
        mapping.get(key): value for key, value in train_dict.items() if value
    }
    print(estimator_params)

    return estimator_params_pt, train_dict_pt


def generate_xgb_training_params(
    framework: str,
    framework_version: str,
    estimator_params: Dict,
    train_dict: Dict,
    sagemaker_session: sagemaker.Session,
) -> Tuple[Dict, Dict]:
    """
    takes in the framework, framework_version, estimator_params and train_dict
    returns a tuple of 2 dictionaries.

    1) contains - framework and framework_version used to retrieve XGBoost image uri
                - all the parameters required by XGBoost Estimator (sagemaker.estimator.Estimator used in mlmonitor).

            {
                leveraging Estimators : https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
                not sagemaker.xgboost.estimator
                XGBOOST specific doc : https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html#create-an-estimator

                "entry_point":  Path (absolute or relative) to the Python source file which should be executed as the entry point to training.
                "source_dir": Path (absolute, relative or an S3 URI) to a directory with any other training source code dependencies aside from the entry point file
                "py_version":  Python version you want to use for executing your model training code
                "framework_version":  XGBoost version you want to use for executing your model training code
                "hyperparameters": Hyperparameters that will be used for training

                EstimatorBase doc : https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html

                "base_job_name": EstimatorBase parameter,
                "role": EstimatorBase parameter,
                "instance_count": EstimatorBase parameter,
                "instance_type": EstimatorBase parameter,
                "output_path": EstimatorBase parameter
            }

    2) contains data location 'training', 'validation', 'testing' (Inconsistent with Scikit-Learn train,test)

    :param framework: str: container image types to use for training
    :param framework_version: str: version of the XGBoost framework used
    :param estimator_params: Dict: estimator parameters
    :param train_dict: Dict: training parameters - might vary from one framework to another
    :param sagemaker_session: sagemaker.Session: instantiated sagemaker
    :return: A tuple of two dictionaries (estimator_params_sk,train_dict_sk)
    """

    return generate_sk_training_params(
        framework, framework_version, estimator_params, train_dict, sagemaker_session
    )


def is_training_job_completed(
    job_name: str, sagemaker_client: Optional[boto_Session.client] = None
) -> bool:
    """
    checks if a training job with the specified name exists.
    It returns True if it does and was completed and False otherwise.

    :param sagemaker_client: Optional Sagemaker client to pass as argument if already instantiated
    :param job_name:str: Pass the name of the training job to filter for
    :return: A boolean value indicating whether the training job exists
    """
    if not sagemaker_client:
        session = boto_Session()
        sagemaker_client = session.client("sagemaker")

    filtered_training_jobs = [
        resource
        for resource in sagemaker_client.list_training_jobs(MaxResults=100).get(
            "TrainingJobSummaries"
        )
        if (
            resource.get("TrainingJobName") == job_name
            and resource.get("TrainingJobStatus") == "Completed"
        )
    ]
    return len(filtered_training_jobs) == 1
