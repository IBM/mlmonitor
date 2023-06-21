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
    json_data = get_feedback_data(access_token, data_mart_id, feedback_dataset_id, url)
    gender_less40_fav_prediction_ratio = 0
    if json_data:
        fields = json_data["records"][0]["fields"]
        print(fields)
        values = json_data["records"][0]["values"]
        import pandas as pd

        feedback_data = pd.DataFrame(values, columns=fields)

        fp = feedback_data.query("Risk == 0 & _original_prediction == 1").shape[0]
        tp = feedback_data.query("Risk == 1 & _original_prediction == 1").shape[0]
        fn = feedback_data.query("Risk == 1 & _original_prediction == 0").shape[0]
        tn = feedback_data.query("Risk == 0 & _original_prediction == 0 ").shape[0]

        female_less40_fav_prediction = len(
            feedback_data.query("Sex == 'female' & Age <= 40 & Risk == 0")
        )
        male_less40_fav_prediction = len(
            feedback_data.query("Sex == 'male' & Age <= 40 & Risk == 0")
        )
        gender_less40_fav_prediction_ratio = (
            female_less40_fav_prediction / male_less40_fav_prediction
        )

    metrics = {
        "specificity": tn / (tn + fp),  # TNR
        "sensitivity": tp / (tp + fn),  # TPR
        "gender_less40_fav_prediction_ratio": gender_less40_fav_prediction_ratio,
        "region": "us-south",
    }

    return metrics
