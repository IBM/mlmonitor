# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import numpy as np
import os
import shutil
import time
from ibm_watson_openscale.supporting_classes.enums import TargetTypes
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import Target

from mlmonitor.src.wos.monitors import get_monitor_id_by_subscription
from mlmonitor.src.wos import wos_client
from mlmonitor.src import DATA_ROOT, MODEL_ROOT
from mlmonitor.src.wos.data_mart import get_datamart_ids
from mlmonitor.src.wos.subscription import get_subscription_id_by_deployment

from mlmonitor.src.model.config import ModelConfig

from ibm_wos_utils.drift.drift_trainer import DriftTrainer


# !TODO get_scores_labels function
def score_aws(training_data_frame: pd.DataFrame) -> tuple:
    """score {function} -- A function that accepts a dataframe with features as columns and returns
    a tuple of numpy array of probabilities array of shape `(n_samples,n_classes)` and numpy array of prediction vector of shape `(n_samples,)`
    Parameters
    ----------
    training_data_frame : pd.DataFrame
       training Dataframe to be used for scoring request to AWS Sagemaker endpoint
    Returns
    -------
    (probabilities,predictions) : tuple
    a tuple of numpy array of probabilities array of shape `(n_samples,n_classes)` and numpy array of prediction vector of shape `(n_samples,)`
    """
    from mlmonitor.src import aws_credentials
    from mlmonitor.src.aws.scoring import _score

    access_id = aws_credentials.get("access_key_id")
    secret_key = aws_credentials.get("secret_access_key")
    region = aws_credentials.get("region")

    pred_json = _score(
        df=training_data_frame,
        access_id=access_id,
        secret_key=secret_key,
        region=region,
        endpoint_name=os.environ["model_endpoint"],
        content_type="csv",
        subscription_id=None,
    )

    predicted_labels = np.array([list(x.values())[0] for x in pred_json])
    proba_scores = np.array([list(x.values())[1] for x in pred_json])

    # proba_vectors = np.array([ [np.round(proba, 3), np.round(1 - proba, 3)] for proba in [list(x.values())[1] for x in pred_json]])

    proba_vectors = [
        list(np.roll(np.array([np.round(proba, 3), np.round(1 - proba, 3)]), label))
        for proba, label in zip(proba_scores, predicted_labels)
    ]

    return np.array(proba_vectors), np.array(predicted_labels)


def score_wml(training_data_frame: pd.DataFrame):
    """score {function} -- A function that accepts a dataframe with features as columns and returns
    a tuple of numpy array of probabilities array of shape `(n_samples,n_classes)` and numpy array of prediction vector of shape `(n_samples,)`
    Parameters
    ----------
    training_data_frame : pd.DataFrame
       training Dataframe to be used for scoring request to WML deployed endpoint
    Returns
    -------
    (probabilities,predictions) : tuple
    a tuple of numpy array of probabilities array of shape `(n_samples,n_classes)` and numpy array of prediction vector of shape `(n_samples,)`
    """
    from mlmonitor.src.wml import wml_client as WML_CLIENT, WML_SPACE_ID
    from mlmonitor.src.wml.scoring import _score

    WML_CLIENT.set.default_space(WML_SPACE_ID)
    pred_json = _score(
        df=training_data_frame,
        endpoint_name=os.environ["model_endpoint"],
        client=WML_CLIENT,
    )

    predicted_labels = [x[0] for x in pred_json.get("values")]
    proba_scores = [x[1] for x in pred_json.get("values")]

    return np.array(proba_scores), np.array(predicted_labels)


def score_azure(training_data_frame: pd.DataFrame) -> tuple:
    """score {function} -- A function that accepts a dataframe with features as columns and returns
    a tuple of numpy array of probabilities array of shape `(n_samples,n_classes)` and numpy array of prediction vector of shape `(n_samples,)`
    Parameters
    ----------
    training_data_frame : pd.DataFrame
       training Dataframe to be used for scoring request to AWS Sagemaker endpoint
    Returns
    -------
    (probabilities,predictions) : tuple
    a tuple of numpy array of probabilities array of shape `(n_samples,n_classes)` and numpy array of prediction vector of shape `(n_samples,)`
    """
    from mlmonitor.src.azure import AZ_WORKSPACE
    from mlmonitor.src.azure.scoring import _score
    from azureml.core.webservice import Webservice

    ws = Webservice(workspace=AZ_WORKSPACE, name=os.environ["model_endpoint"])

    pred_json = _score(
        df=training_data_frame,
        workspace=AZ_WORKSPACE,
        endpoint_name=os.environ["model_endpoint"],
        content_type="json",
        compute_type=ws.compute_type.lower(),
    )

    predicted_labels = [list(x.values())[0] for x in pred_json.get("output")]
    proba_scores = [list(x.values())[1] for x in pred_json.get("output")]

    return np.array(proba_scores), np.array(predicted_labels)


def configure_drift(
    model_config: ModelConfig,
    deployment_name: str,
    deployment_target: str = "aws",
    keep_wos_monitor: bool = True,
    data_path: str = DATA_ROOT,
    data_mart_id: str = None,
) -> dict:
    """Configure a drift monitor for the deployed Sagemaker Endpoint with name <deployment_name>
    Parameters
    ----------
    model_config : str
       ModelConfig: Configuration parameters for the quality monitor
    deployment_name : str
       EndpointName of AWS or WML online inference endpoint.
    deployment_target : str
       target model serving environment
       wml , aws , azure , custom
    keep_wos_monitor : bool
       this boolean flag indicates whether an existing OpenScale Fairness monitor for  deployment_name should be kept or re-created
    data_path : str
       directory Path where train and validation dataset should be located. default is Project datasets folder
    data_mart_id : str
       Data Mart ID
    Returns
    -------
    drift_monitor_instance_id : str
    Watson OpenScale Fairness Monitor ID created or found (if keep_wos_monitor=True) for the input deployment_name
    """

    assert deployment_target in {
        "wml",
        "aws",
        "azure",
    }, "Deployment target should be wml , aws or azure"

    training_data_file_name = os.path.join(data_path, model_config.training_data)
    data_df = pd.read_csv(training_data_file_name, engine="python")
    # drift_archive_path = os.path.join(MODEL_ROOT, 'drift_detection_model_churn.tar.gz')
    drift_archive_path = os.path.join(
        MODEL_ROOT, f"drift_detect_mdl_{model_config.source_dir}.tar.gz"
    )

    if os.path.exists(drift_archive_path):
        os.remove(drift_archive_path)

    drift_parameters = model_config.drift_monitor_parameters

    subscription_ids = get_subscription_id_by_deployment(
        wos_client=wos_client, deployment_name=deployment_name
    )

    if len(subscription_ids) != 1:
        raise ValueError(
            f"{deployment_name} should have exactly one subscription ID => {len(subscription_ids)} found"
        )

    if not data_mart_id:
        data_marts = get_datamart_ids(wos_client=wos_client)

        if len(data_marts) != 1:
            raise ValueError(f"Please Specify datamart to use among {data_marts}")

        data_mart_id = data_marts[0]

    subscription_id = subscription_ids[0]
    # Find Monitors in place for a given SUBSCRIPTION_ID
    drift_monitor_instance_id = get_monitor_id_by_subscription(
        wos_client=wos_client, subscription_id=subscription_id, monitor_type="drift"
    )

    if not keep_wos_monitor and drift_monitor_instance_id:
        wos_client.monitor_instances.delete(
            monitor_instance_id=drift_monitor_instance_id, background_mode=False
        )
        drift_monitor_instance_id = None

    if not drift_monitor_instance_id:
        print(f"No drift monitor for {deployment_name}")
        drift_detection_input = {
            "feature_columns": model_config.feature_columns,
            "categorical_columns": model_config.categorical_columns,
            "label_column": model_config.class_label,
            "problem_type": model_config.problem_type,
        }

        if deployment_target == "aws":
            proba_arr, pred_vector = score_aws(
                training_data_frame=data_df.loc[0:5, model_config.feature_columns]
            )
        elif deployment_target == "wml":
            proba_arr, pred_vector = score_wml(
                training_data_frame=data_df.loc[0:5, model_config.feature_columns]
            )
        elif deployment_target == "azure":
            proba_arr, pred_vector = score_azure(
                training_data_frame=data_df.loc[0:5, model_config.feature_columns]
            )
        else:
            raise NotImplementedError("deployment_target should be aws wml or azure")

        print(proba_arr, pred_vector)

        drift_trainer = DriftTrainer(data_df, drift_detection_input)
        if model_config.problem_type == "regression":
            raise ValueError("drift not supported for regression problem type")

        elif not os.path.exists(drift_archive_path):
            # Note: batch_size can be customized by user as per the training data size
            # drift_trainer.generate_drift_detection_model(score, batch_size=data_df.shape[0], check_for_ddm_quality=False)

            if deployment_target == "aws":
                drift_trainer.generate_drift_detection_model(
                    score=score_aws, batch_size=data_df.shape[0], optimise=True
                )
            elif deployment_target == "wml":
                drift_trainer.generate_drift_detection_model(
                    score=score_wml, batch_size=data_df.shape[0], optimise=True
                )
            elif deployment_target == "azure":
                drift_trainer.generate_drift_detection_model(
                    score=score_azure, batch_size=data_df.shape[0], optimise=True
                )
            else:
                raise NotImplementedError("deployment_target should be aws or wml")

            drift_trainer.learn_constraints(**model_config.drift_learn_constraints)
            drift_trainer.create_archive()
            shutil.move("drift_detection_model.tar.gz", drift_archive_path)

        else:
            print(
                f"Drift archive {drift_archive_path} already exists Skipping generation.."
            )

        wos_client.monitor_instances.upload_drift_model(
            model_path=drift_archive_path,
            data_mart_id=data_mart_id,
            subscription_id=subscription_id,
        )

        drift_monitor_details = wos_client.monitor_instances.create(
            data_mart_id=data_mart_id,
            background_mode=False,
            monitor_definition_id=wos_client.monitor_definitions.MONITORS.DRIFT.ID,
            target=Target(
                target_type=TargetTypes.SUBSCRIPTION, target_id=subscription_id
            ),
            parameters=drift_parameters,
        ).result

        drift_monitor_instance_id = drift_monitor_details.metadata.id
        print(f"drift Monitor ID Created [{drift_monitor_instance_id}]")
        time.sleep(5)
        wos_client.monitor_instances.show_metrics(
            monitor_instance_id=drift_monitor_instance_id
        )

    return drift_monitor_instance_id
