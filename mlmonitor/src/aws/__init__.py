# SPDX-License-Identifier: Apache-2.0
from sagemaker.sklearn.estimator import SKLearnModel  # , SKLearn
from sagemaker.xgboost import XGBoostModel  # , XGBoost
from sagemaker.tensorflow import TensorFlowModel, TensorFlow
from sagemaker.pytorch import PyTorchModel, PyTorch
from sagemaker.estimator import Estimator

from sagemaker.deserializers import CSVDeserializer
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer

# https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html
# https://github.com/aws/sagemaker-python-sdk#installing-the-sagemaker-python-sdk
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
sagemaker_models = {
    "sklearn": SKLearnModel,
    "xgboost": XGBoostModel,
    "tensorflow": TensorFlowModel,
    "pytorch": PyTorchModel,
}
sagemaker_estimators = {
    "sklearn": Estimator,
    "xgboost": Estimator,
    "tensorflow": TensorFlow,
    # https://sagemaker.readthedocs.io/en/v2.8.0/frameworks/tensorflow/sagemaker.tensorflow.html#tensorflow-estimator
    "pytorch": PyTorch,
}
sagemaker_serializers = {"json": JSONSerializer, "csv": CSVSerializer}
sagemaker_deserializers = {"json": JSONDeserializer, "csv": CSVDeserializer}

SUPPORTED_SAGEMAKER_ESTIMATORS = list(sagemaker_estimators.keys())
