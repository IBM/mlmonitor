# SPDX-License-Identifier: Apache-2.0
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import xgboost as xgb

sns.set_style("whitegrid")


def plot_confusion_matrix(
    cm,
    normalize=None,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    prefix="xgboost",
    dir: str = "./",
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    if normalize is not None:
        types = {"recall": 1, "precision": 0}
        cm = cm.astype("float") / cm.sum(axis=types[normalize])[:, np.newaxis]
        print("Normalized confusion matrix {0}".format(normalize))
    else:
        print("Confusion matrix, without normalization")

    plt.subplots(figsize=(5, 3))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm, cmap=cmap, annot=True, xticklabels=[0, 1], yticklabels=[0, 1], fmt="g"
    )

    plt.title(title)
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.savefig(f"{dir}/{prefix}_confusion_matrix.png", bbox_inches="tight")
    plt.show()


def plot_precision_recall_vs_threshold(
    precisions, recalls, thresholds, prefix="xgboost", dir: str = "./"
):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc="best")
    plt.savefig(f"{dir}/{prefix}_pr_curve.png", bbox_inches="tight")
    plt.show()


def plot_roc_curve(fpr, tpr, label=None, dir: str = "./"):
    """
    The ROC curve, modified from
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8, 8))
    plt.title("ROC Curve")
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="best")
    plt.savefig(f"{dir}/xgboost_roc_curve.png", bbox_inches="tight")
    plt.show()


def custom_pred_distro(
    positives, negatives, cutoff=0.5, title="xgboost", dir: str = "./"
):
    """This function generates distributions of predicted scores for actual positives and actual negatives.
    Note that the cutoff argument only affects the coloring of the graphs. It does NOT affect any model
    results or predicted values.
    Source : https://github.com/aws-samples/amazon-sagemaker-custom-loss-function
    """

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    set_ax_definition(axes, 0, "Actual Negatives")
    axes[0].hist(
        negatives[negatives > cutoff], color="C1", label="False Positives", bins=30
    )
    axes[0].hist(negatives[negatives <= cutoff], label="True Negatives", bins=30)
    axes[0].legend()

    set_ax_definition(axes, 1, "Actual Positives")
    axes[1].hist(positives[positives > cutoff], label="True Positives", bins=30)
    axes[1].hist(positives[positives <= cutoff], label="False Negatives", bins=30)
    axes[1].legend()

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold", x=0.52)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.tight_layout()

    plt.savefig(f"{dir}/{title}_probabilities.png", bbox_inches="tight")


def set_ax_definition(axes, idx, title):
    axes[idx].spines["top"].set_visible(False)
    axes[idx].spines["right"].set_visible(False)
    axes[idx].set(
        xlim=[0, 1],
        xticks=np.arange(0, 1, step=0.1),
        xlabel="Model Score",
        ylabel="Count",
        title=title,
    )


def plot_history(
    model: xgb.XGBClassifier, filename: str, eval_metric: str, dir: str = "./"
):
    """plot eval_metric on test set during train at each epoch
    This method allows save a plot of training eval_metric over time (epoch)
    Parameters
    ----------
    model : xgb.XGBClassifier
        trained XGB model
    filename : str
        filename of the figure to be saved
    eval_metric : str
        evaluation metric to be displayed for validation data
    Returns
    -------
    """
    fig, ax = plt.subplots()
    results = model.evals_result()
    epochs = len(results["validation_0"][eval_metric])
    x_axis = range(epochs)
    pd.DataFrame(model.evals_result()["validation_0"]).plot()
    ax.plot(x_axis, results["validation_0"][eval_metric])
    ax.plot(x_axis, results["validation_1"][eval_metric])
    ax.set_title(f"xgboost model training {eval_metric} over epochs")
    ax.set_xlabel("epochs")
    ax.set_ylabel(eval_metric)
    ax.legend(["train", "val"])
    fig.savefig(os.path.join(dir, filename))
    plt.show()


def plot_feature_importance(mdl: xgb.XGBRegressor, dir: str):
    plt.figure(figsize=(15, 15))
    xgb.plot_importance(mdl, height=0.9, max_num_features=100)
    plt.title("feature importance for churn prediction model")
    plt.savefig(f"{dir}/xgboost_feature_importance.png", bbox_inches="tight")
    plt.show()
