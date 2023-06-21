# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks


def train_model(args):
    """Generate a simple model"""

    x_train, y_train = _load_training_data(args.get("train"))
    x_test, y_test = _load_testing_data(args.get("test"))

    print(x_train.shape)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )

    earlystopping = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.get("epochs"),
        batch_size=args.get("batch_size"),
        callbacks=[earlystopping],
    )
    model.evaluate(x_test, y_test)

    if args.get("current_host") == args.get("hosts")[0]:
        # save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format
        # To export the model as h5 format use model.save('my_model.h5')
        model.save(os.path.join(args.get("model_dir"), "000000001"))

    return model


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
    # Training Job specific arguments (Sagemaker,Azure,WML) default SageMaker envar or Azure expected values
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./outputs"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
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
    mnist_classifier = train_model(args)
