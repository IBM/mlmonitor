# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score,
)
import xgboost as xgb

try:
    from visualize import (
        custom_pred_distro,
        plot_confusion_matrix,
    )
except ImportError as e:
    print(
        f"use_case_churn.metrics could not import modules => not running in AWS job : {e}"
    )
    from mlmonitor.use_case_churn.visualize import (
        custom_pred_distro,
        plot_confusion_matrix,
    )


def assign_classes(df_proba: pd.DataFrame, threshold: float) -> np.array:
    """outputs class predictions based on predicted probabilities and a given threshold t
        for binary classification problems,this method assigns the class predictions based on a probability a a threshold (t).
    Parameters
    ----------
    df_proba : pd.DataFrame
        predicted probabilities for each class
    threshold : pd.DataFrame
        probability threshold between zero and one above which class 1 (positive) will be assigned
    Returns
    -------
    assigned_classes : np.array
        output data of shape N with assigned predicted classes
    """
    print(df_proba.columns)
    if "1_proba" not in df_proba.columns:
        raise ValueError("1_proba column should be in dataframe columns")

    assigned_classes = np.where(df_proba["1_proba"].values > threshold, 1, 0)
    return assigned_classes


def precision(predictions: np.array, ground_truth: np.array) -> float:
    """precision score
        for binary classification problems returns the precision score TP/(TP+FP)
    Parameters
    ----------
    predictions : np.array
        predicted classes for each datapoint
    ground_truth : np.array
       ground truth value for each datapoint
    Returns
    -------
    precision : float
        computed precision score
    """
    tp = np.sum(np.logical_and(predictions == 1, ground_truth == 1))
    fp = np.sum(np.logical_and(predictions == 1, ground_truth == 0))

    precision = tp / (tp + fp)

    return precision


def recall(predictions: np.array, ground_truth: np.array) -> float:
    """recall score
        for binary classification problems returns the recall score TP/(TP+FN)
    Parameters
    ----------
    predictions : np.array
        predicted classes for each datapoint
    ground_truth : np.array
       ground truth value for each datapoint
    Returns
    -------
    recall : float
        computed recall score
    """
    tp = np.sum(np.logical_and(predictions == 1, ground_truth == 1))
    fn = np.sum(np.logical_and(predictions == 0, ground_truth == 1))

    recall = tp / (tp + fn)

    return recall


def false_positive_rate(predictions: np.array, ground_truth: np.array) -> float:
    """false positive rate
        for binary classification problems returns the false positive rate score FP/(TN+FP)
        What is the ratio of false alarms of all the negative examples (non failure)
    Parameters
    ----------
    predictions : np.array
        predicted classes for each datapoint
    ground_truth : np.array
       ground truth value for each datapoint
    Returns
    -------
    fpr : float
        computed false positive rate
    """
    fp = np.sum(np.logical_and(predictions == 1, ground_truth == 0))
    fn = np.sum(np.logical_and(predictions == 0, ground_truth == 1))

    fpr = fp / (fp + fn)

    return fpr


def eval_model(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    y_test_pred: pd.DataFrame,
    threshold: float = 0.5,
    local: bool = False,
    dir: str = "./",
) -> tuple:

    df_proba = pd.DataFrame(y_test_pred, columns=["0_proba", "1_proba"])

    y_test_pred2 = assign_classes(df_proba=df_proba, threshold=threshold)
    print("proba :", Counter(df_proba["1_proba"]).most_common(3))

    df_test_predictions = x_test.copy()
    df_test_predictions["labels"] = y_test
    df_test_predictions["proba"] = df_proba["1_proba"].values

    positives = df_test_predictions.loc[
        df_test_predictions["labels"] == 1, "proba"
    ].values
    negatives = df_test_predictions.loc[
        df_test_predictions["labels"] == 0, "proba"
    ].values

    acc_test = accuracy_score(y_test, y_test_pred2)
    recall_test = recall_score(y_test, y_test_pred2, labels=[1], average="macro")
    prec_test = precision_score(y_test, y_test_pred2, labels=[1], average="macro")
    f1_test = f1_score(y_test, y_test_pred2, labels=[1], average="macro")
    CM = confusion_matrix(y_test, y_test_pred2)

    tp = np.sum(np.logical_and(y_test_pred2 == 1, y_test == 1))
    tn = np.sum(np.logical_and(y_test_pred2 == 0, y_test == 0))
    fp = np.sum(np.logical_and(y_test_pred2 == 1, y_test == 0))
    fn = np.sum(np.logical_and(y_test_pred2 == 0, y_test == 1))

    prec = precision(predictions=y_test_pred2, ground_truth=y_test)
    rec = recall(predictions=y_test_pred2, ground_truth=y_test)
    fpr = false_positive_rate(predictions=y_test_pred2, ground_truth=y_test)

    print(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
    print(f"Confusion Matrix \n{CM}")
    print(f"precision {prec} / recall {rec}")
    print(f"FPR {fpr}")
    print(f"f1 {2 * prec * rec / (prec + rec)} ")

    print(f"F1  test : {f1_test}")
    print(f"Acc test : {acc_test}")
    print(f"Prec test : {prec_test}")
    print(f"Rec test : {recall_test}")

    tags = {"xgboost_version": xgb.__version__, "Confusion Matrix": str(CM)}

    metrics = {
        "prec": prec,
        "acc_test": acc_test,
        "recall_test": recall_test,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "f1": 2 * prec * rec / (prec + rec),
        "fn": fn,
        "rec": rec,
        "fpr": fpr,
        "F1  test": f1_test,
        "Acc test": acc_test,
        "Prec test": prec_test,
        "Rec test": recall_test,
    }
    print(metrics)
    params = {"threshold": threshold}

    if local:
        custom_pred_distro(
            positives, negatives, cutoff=threshold, title="xgboost", dir=dir
        )
        plot_confusion_matrix(cm=CM, prefix="xgboost", dir=dir)

    return metrics, params, tags
