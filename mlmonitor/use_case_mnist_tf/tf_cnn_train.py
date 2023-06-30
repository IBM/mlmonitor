# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
import json
import time
import os
import numpy as np
from typing import Optional

from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam  # Adadelta
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential

from tensorflow.keras import backend as keras_backend

try:
    from tf_models import base_model
    from factsheets_helpers import init_external_fs_client, save_fs_model
    from utils import init_logger, generate_data
except ImportError as e:
    print(
        f"use_case_mnist_tf.tf_cnn_train could not import modules => not running in AWS job : {e}"
    )
    from mlmonitor.use_case_mnist_tf.tf_models import base_model
    from mlmonitor.use_case_mnist_tf.factsheets_helpers import (
        init_external_fs_client,
        save_fs_model,
    )
    from mlmonitor.use_case_mnist_tf.utils import init_logger, generate_data

num_classes = 10
log_level = int(os.getenv("LOG_LEVEL", logging.INFO))

logger = init_logger(level=log_level)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_wml(
    model_dir: str,
    data_path: str,
    train_dataset: str,
    val_dataset: Optional[str] = None,
    test_dataset: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **hyperparameters,
) -> str:
    """
    train_wml is used to train this model in local environment
    this python module `use_case_gcr` is imported dynamically by `mlmonitor`

    - this function definition should always contain as input parameters :
    model_dir , data_path , train_dataset,val_dataset,test_dataset ,logger ,and  hyperparameters as parameters

    - this function must produce a model artifact return its location in model_data pah

    .. code-block:: python
        :caption: Example
        from mlmonitor import WMLModelUseCase
        model_use_case = WMLModelUseCase(source_dir='use_case_gcr', catalog_id=catalog_id, model_entry_id=model_entry_id)
        model_use_case.train() => this function is invoked by trained task

    :param model_dir:str: Base directory where to store model after training completion
    :param data_path:str: location (directory path) of the datasets for this model use case
    :param train_dataset:str: filename of training dataset
    :param val_dataset:Optional[str]=None:  filename of validation dataset
    :param test_dataset:Optional[str]=None:  filename of test dataset
    :param logger:Optional[logging.Logger]=None: Pass instantiated logger object
    :param **hyperparameters: model hyperparameters to use for model training task
    :return: path to the model artifact produced
    """

    trained_model = train_loop(
        train_data=data_path,
        test_data=data_path,
        logger=logger,
        **{
            "epochs": hyperparameters.get("epochs"),
            "batch_size": hyperparameters.get("batch-size"),
        },
    )

    model_data = os.path.join(model_dir, "model_mnist", "mnist_cnn_aws.h5")
    trained_model.save(model_data)

    return model_data


def train_loop(
    train_data: str,
    test_data: str,
    logger: logging.Logger,
    **hyperparameters,
) -> Sequential:
    img_rows, img_cols = 28, 28
    verbose = 0 if os.environ.get("SM_TRAINING_ENV") else 1

    if channels_first := (keras_backend.image_data_format() == "channels_first"):
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    x_train, y_train = generate_data(
        data_path=train_data, data_type="train", channel_first=channels_first
    )
    x_test, y_test = generate_data(
        data_path=test_data, data_type="test", channel_first=channels_first
    )

    logger.info(f"x_train shape: {x_train.shape}")
    logger.info(f"x_test shape: {x_test.shape}")

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    cnn_n = base_model(input_shape=input_shape)
    cnn_n.summary()

    earlystopping = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True
    )

    cnn_n.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"])

    cnn_n.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=hyperparameters.get("epochs"),
        batch_size=hyperparameters.get("batch_size"),
        callbacks=[earlystopping],
        verbose=verbose,
    )

    scores = cnn_n.evaluate(x_test, y_test, verbose=0)
    logger.info(scores)
    logger.info("Accuracy: %.2f%%" % (scores[1] * 100))

    return cnn_n


def train_job(args):
    """Generate a simple model"""
    # Init Factsheet Client
    (
        facts_client,
        props,
        EXPERIMENT_NAME,
        EXPERIMENT_ID,
        tags,
        params,
    ) = init_external_fs_client(
        logger=logger,
        ibm_key_name=args.get("ibm_key_name"),
        region_name=args.get("region_name"),
        catalog_id=args.get("catalog_id"),
        model_entry_id=args.get("model_entry_id"),
        cp4d_env=args.get("cp4d_env"),
        cp4d_username=args.get("cp4d_username"),
        cp4d_url=args.get("cp4d_url"),
    )

    start = time.time()

    trained_model = train_loop(
        train_data=args.get("train"),
        test_data=args.get("test"),
        logger=logger,
        **{"epochs": args.get("epochs"), "batch_size": args.get("batch_size")},
    )

    if args.get("current_host") == args.get("hosts")[0]:
        # save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format
        # To export the model as h5 format use model.save('my_model.h5')
        trained_model.save(os.path.join(args.get("model_dir"), "000000001"))
        trained_model.save(os.path.join(args.get("model_dir"), args.get("model_name")))

    end = time.time()
    metrics = {"train_duration_sec": np.round(end - start, 4)}

    save_fs_model(
        logger=logger,
        facts_client=facts_client,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        catalog_id=args.get("catalog_id"),
        model_entry_id=args.get("model_entry_id"),
        tags=tags,
        params=params,
        metrics=metrics,
    )

    return os.path.join(args.get("model_dir"), args.get("model_name"))


def _load_training_data(base_dir):
    """Load MNIST training data"""
    x_train = np.load(os.path.join(base_dir, "train_data.npy"))
    y_train = np.load(os.path.join(base_dir, "train_labels.npy"))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load MNIST testing data"""
    x_test = np.load(os.path.join(base_dir, "eval_data.npy"))
    y_test = np.load(os.path.join(base_dir, "eval_labels.npy"))
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    # fmt: off
    # CP4D specific arguments
    parser.add_argument("--catalog-id", type=str)
    parser.add_argument("--model-entry-id", type=str)
    parser.add_argument("--ibm-key-name", type=str, default="IBM_API_KEY_MLOPS")
    parser.add_argument("--cp4d-env", type=str, default=os.getenv("ENV", "saas"), choices=["saas", "prem"])
    parser.add_argument("--cp4d-username", type=str, default=None)
    parser.add_argument("--cp4d-url", type=str, default=None)

    # Training Job specific arguments (Sagemaker,Azure,WML) default SageMaker envar or Azure expected values
    parser.add_argument("--model-name", type=str, default='mnist_cnn_aws.h5')
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./outputs"))
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAINING"))
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TESTING"))
    parser.add_argument("--validation", type=str, default=os.getenv("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.getenv("SM_HOSTS", '["algo-1"]')))
    parser.add_argument("--current-host", type=str, default=os.getenv("SM_CURRENT_HOST", "algo-1"))
    parser.add_argument("--region-name", type=str, default="ca-central-1")

    # Model specific hyperparameters
    parser.add_argument("--batch-size", type=int, default=128, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=3, metavar="N", help="number of epochs to train (default: 1)")
    # fmt: on
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    args = vars(args)
    mnist_classifier = train_job(args)
