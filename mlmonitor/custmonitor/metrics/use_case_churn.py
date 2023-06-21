# SPDX-License-Identifier: Apache-2.0
from custmonitor.metricsprovider.helpers import (
    get_feedback_dataset_id,
    get_feedback_data,
)


def get_metrics(access_token, data_mart_id, subscription_id, url):
    # Add the logic here to compute the metrics. Use the below metric names while creating the custom monitor definition
    feedback_dataset_id = get_feedback_dataset_id(
        access_token, data_mart_id, subscription_id, url
    )

    if json_data := get_feedback_data(
        access_token, data_mart_id, feedback_dataset_id, url
    ):
        fields = json_data["records"][0]["fields"]
        print(fields)
        values = json_data["records"][0]["values"]
        import pandas as pd
        import numpy as np

        feedback_data = pd.DataFrame(values, columns=fields)

        tp = np.sum(
            np.logical_and(
                feedback_data["_original_prediction"].values == 1,
                feedback_data["Churn?_True."].values == 1,
            )
        )
        fp = np.sum(
            np.logical_and(
                feedback_data["_original_prediction"].values == 1,
                feedback_data["Churn?_True."].values == 0,
            )
        )
        tn = np.sum(
            np.logical_and(
                feedback_data["_original_prediction"].values == 0,
                feedback_data["Churn?_True."].values == 0,
            )
        )
        fn = np.sum(
            np.logical_and(
                feedback_data["_original_prediction"].values == 0,
                feedback_data["Churn?_True."].values == 1,
            )
        )

        return {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "cost": int(500 * fn + 0 * tn + 100 * fp + 100 * tp),
            "total": int(fn + tn + fp + tp),
        }

    else:
        return {}
