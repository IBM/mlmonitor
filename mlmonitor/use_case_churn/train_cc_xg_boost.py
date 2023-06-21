# SPDX-License-Identifier: Apache-2.0
import json
import re
import pandas as pd
import os
import logging
import xgboost as xgb
from typing import Optional
import joblib
from os.path import dirname, abspath

try:
    from utils import _init_logger, _parse_args, save_model
    from metrics import eval_model
    from factsheets import init_external_fs_client, save_fs_model
    from visualize import plot_history
except ImportError as e:
    print(
        f"use_case_churn.train_cc_xg_boost could not import modules => not running in AWS job : {e}"
    )
    from mlmonitor.use_case_churn.utils import _init_logger, _parse_args, save_model
    from mlmonitor.use_case_churn.metrics import eval_model
    from mlmonitor.use_case_churn.factsheets import (
        init_external_fs_client,
        save_fs_model,
    )
    from mlmonitor.use_case_churn.visualize import plot_history

logger = _init_logger()


def fetch_dataset(data_path: str, file_type: str = "train") -> pd.DataFrame:
    """
    takes a directory of CSV files and concatenates them into a single Pandas dataframe.
    It assumes that the files contain comma-separated values with no header row.
    The function accepts an optional file_type parameter, which can be used to filter files to include.

    :param data_path:str:  data location (valid directory path)
    :param file_type:str: Specify whether the data is train or test or validation
    :return: pandas dataframe
    """
    # Take the set of files and read them all into a single pandas dataframe
    pattern = re.compile(rf"{file_type}.*\.csv")
    input_files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if pattern.match(file)
    ]
    logger.info(input_files)
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(data_path, "train")
        )

    raw_data = [pd.read_csv(file, engine="python") for file in input_files]

    return pd.concat(raw_data)


class XGBModel:
    def __init__(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, hparams: dict = {}
    ):
        """
        :param train_data:pd.DataFrame: Pass the training data
        :param test_data:pd.DataFrame: Evaluate the model
        :param hparams:dict={}: Pass in the hyperparameters
        """
        # train_data
        self.x_train = train_data.iloc[:, 1:]
        self.y_train = train_data.iloc[:, 0]
        self.x_test = test_data.iloc[:, 1:]
        self.y_test = test_data.iloc[:, 0]
        self.eval_metric = "logloss"

        negative_examples = train_data.iloc[:, 0].value_counts().loc[0]
        positive_examples = test_data.iloc[:, 0].value_counts().loc[1]
        self.scale_pos_weight = int(negative_examples / positive_examples)

        if len(hparams) == 0:

            logger.info(
                f"negative_examples: {negative_examples}\n"
                f"positive_examples : {positive_examples}\n"
                f"scale_pos_weight : {self.scale_pos_weight}"
            )

            self.params = {
                "max_depth": 5,
                "n_estimators": 200,
                "booster": "gbtree",
                "learning_rate": 0.001,
                "scale_pos_weight": self.scale_pos_weight,
                "eta": 0.2,
                "gamma": 4,
                "min_child_weight": 6,
                "subsample": 0.8,
                "objective": "binary:logistic",
                "num_round": 200,
            }
        else:
            # https://xgboost.readthedocs.io/en/stable/parameter.html
            self.params = hparams
            self.params["learning_rate"] = self.params["eta"]
            self.params["sampling_method"] = "uniform"
            self.params["reg_lambda"] = 1
            self.params["reg_alpha"] = 0
            self.params["scale_pos_weight"] = self.scale_pos_weight
            self.params["grow_policy"] = "depthwise"
            self.params["max_leaves"] = 0
            self.params["tree_method"] = "approx"

        self.model = xgb.XGBClassifier(**self.params)
        self.number_trees = self.model.get_params().get("n_estimators")
        logger.info(f"xgb.XGBRegressor with parameters :{self.model.get_params()}")

    def fit(self) -> xgb.XGBClassifier:
        self.model.fit(
            X=self.x_train,
            y=self.y_train,
            eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
            eval_metric=[self.eval_metric],
            early_stopping_rounds=int(self.number_trees * 0.1),
        )
        return None

    def predict(self, data):
        res = self.model.predict(data)
        return res


def train_wml(
    model_dir: str,
    data_path: str,
    train_dataset: str,
    val_dataset: Optional[str] = None,
    test_dataset: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **hyperparameters,
):
    """
    train_wml is used to train this model in local environment
    this python module `use_case_churn` is imported dynamically by `mlmonitor`

    - this function definition should always contain as input parameters :
    model_dir , data_path , train_dataset,val_dataset,test_dataset ,logger ,and  hyperparameters as parameters

    - this function must produce a model artifact return its location in model_data pah

    .. code-block:: python
        :caption: Example
        from mlmonitor import WMLModelUseCase
        model_use_case = WMLModelUseCase(source_dir='use_case_churn', catalog_id=catalog_id, model_entry_id=model_entry_id)
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
    FIGURES_ROOT = abspath(os.path.join(dirname(dirname(__file__)), "figures"))
    logger.info(
        f"train_wml function fit XGBModel with params\n:{json.dumps(hyperparameters, indent=4)}"
    )
    train_data = fetch_dataset(data_path=data_path, file_type="train")
    val_data = fetch_dataset(data_path=data_path, file_type="validation")
    model = XGBModel(train_data=train_data, test_data=val_data, hparams=hyperparameters)
    model.fit()

    y_test_pred = model.model.predict_proba(model.x_test)

    metrics, params, tags = eval_model(
        x_test=model.x_test,
        y_test=model.y_test,
        y_test_pred=y_test_pred,
        local=True,
        dir=FIGURES_ROOT,
    )

    joblib.dump(model.model, os.path.join(model_dir, "model_churn", "model.joblib"))
    return os.path.join(model_dir, "model_churn", "model.joblib")


def run_train_job(logger, local=False, save=True, **parameters) -> XGBModel:
    """
    run_train_job function trains churn prediction model model and saves it to disk.
    it is invoked in a Sagemaker training job

    - instantiates Factheets client for external models
    - fetches training and validation datasets
    - trains Xgboost model with specified hyperparameters and display figures is model trained locally (loss,confusion matrix)
    - Log training metrics to AI Factsheets with relevant model parameters ands tags

    :param logger: Log the training process
    :param local=False: whether the training is happening in sagemaker or locally
    :param save=True: whether to Save the model to disk
    :param **parameters: hyperparameters that are used to train the model
    :return: An xgbmodel object
    """
    xgb_version = xgb.__version__
    logger.info(f"XGBOOST VERSION {xgb_version}")

    (
        facts_client,
        props,
        EXPERIMENT_NAME,
        EXPERIMENT_ID,
        fs_tags,
        fs_params,
    ) = init_external_fs_client(logger=logger, **parameters)
    fs_metrics = {}

    hyperparameters = {
        "max_depth": parameters.get("max_depth"),
        "eta": parameters.get("eta"),
        "gamma": parameters.get("gamma"),
        "min_child_weight": parameters.get("min_child_weight"),
        "subsample": parameters.get("subsample"),
        "objective": parameters.get("objective"),
        "num_round": parameters.get("num_round"),
        "verbosity": parameters.get("verbosity"),
        "n_estimators": parameters.get("n_estimators"),
    }

    fs_params = {**fs_params, **hyperparameters}

    train_data = fetch_dataset(data_path=parameters.get("train"), file_type="train")
    val_data = fetch_dataset(
        data_path=parameters.get("validation"), file_type="validation"
    )

    model = XGBModel(train_data=train_data, test_data=val_data, hparams=hyperparameters)
    model.fit()
    y_test_pred = model.model.predict_proba(model.x_test)

    metrics, params, tags = eval_model(
        x_test=model.x_test,
        y_test=model.y_test,
        y_test_pred=y_test_pred,
        local=local,
        dir=parameters.get("output_data_dir"),
    )

    fs_metrics = {**fs_metrics, **metrics}
    fs_tags = {**fs_tags, **tags}
    fs_params = {**fs_params, **params}

    save_fs_model(
        logger=logger,
        facts_client=facts_client,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        catalog_id=parameters.get("catalog_id"),
        model_entry_id=parameters.get("model_entry_id"),
        inputs=None,
        outputs=None,
        tdataref=None,
        tags=fs_tags,
        params=fs_params,
        metrics=fs_metrics,
    )

    if save:
        save_model(
            model=model,
            dir=parameters.get("model_dir"),
            name=parameters.get("model_name"),
        )
        logger.info(f"Stored trained model at {parameters.get('model_dir')}")

    if local:
        from visualize import plot_feature_importance  # , plot_history

        plot_history(
            model.model,
            filename=f"{parameters.get('output_data_dir')}/loss_xgb_notebook.png",
            eval_metric="logloss",
            dir=parameters.get("output_data_dir"),
        )
        plot_feature_importance(mdl=model.model, dir=parameters.get("output_data_dir"))

    return model


if __name__ == "__main__":
    args, unknown = _parse_args()
    parameters = vars(args)
    run_train_job(logger=logger, local=False, **parameters)
