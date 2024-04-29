# SPDX-License-Identifier: Apache-2.0
import os
import torch
import time
import sys
import shutil
import json
import logging
from pygit2 import Repository
from pygit2 import GitError
import numpy as np
from ibm_aigov_facts_client import AIGovFactsClient, CloudPakforDataConfig

from pytorch_lightning import Trainer
from torchinfo import summary
from utils import parse_args, get_secret
from use_case_mnist_ptlt.pt_models import (
    PytorchLightningMNIST,
    PytorchLightning_CNN_MNIST,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(arguments):

    if os.environ.get("SM_TRAINING_ENV"):
        # verbose = 0
        SM_TRAINING_ENV = json.loads(os.environ["SM_TRAINING_ENV"])
        EXPERIMENT_NAME = SM_TRAINING_ENV["job_name"]
        logger.info(f"Running in a SM JOB {EXPERIMENT_NAME}")
        # install_dependencies()
        API_KEY_NAME = arguments.get("ibm_key_name")
        region_name = arguments.get("region_name", "ca-central-1")
        API_KEY = json.loads(
            get_secret(secret_name="IBM_KEYS", region_name=region_name)
        ).get(API_KEY_NAME)
    else:
        time_suffix = time.strftime("%Y%m%d-%H%M", time.gmtime())
        EXPERIMENT_NAME = f"sm-mnist-{arguments.get('model_type')}-ptlt-{time_suffix}"
        API_KEY = os.getenv("API_KEY")

    try:
        repo = Repository(PROJECT_ROOT)
        branch = repo.head.raw_shorthand.decode("utf-8")
    except GitError as e:
        logger.warning(f"GitError {e}")
        branch = "Not Found"

    # Init Factsheet Client
    start = time.time()

    if arguments.get("cp4d_env") == "saas":
        facts_client = AIGovFactsClient(
            api_key=API_KEY,
            experiment_name=EXPERIMENT_NAME,
            external_model=True,
            enable_autolog=True,
            set_as_current_experiment=True,
        )
    elif arguments.get("cp4d_env") == "prem":
        facts_client = AIGovFactsClient(
            cloud_pak_for_data_configs=CloudPakforDataConfig(
                service_url=arguments.get("cp4d_url"),
                username=arguments.get("cp4d_username"),
                api_key=API_KEY,
            ),
            experiment_name=EXPERIMENT_NAME,
            external_model=True,
            enable_autolog=True,
            set_as_current_experiment=True,
        )

    # get experiment_id
    EXPERIMENT_ID = facts_client.experiments.get_current_experiment_id()
    logger.info(f"Current Experiment {EXPERIMENT_NAME} ID {EXPERIMENT_ID}")

    if arguments.get("model_type") == "fc":
        model = PytorchLightningMNIST(arguments)
    elif arguments.get("model_type") == "cnn":
        model = PytorchLightning_CNN_MNIST(arguments)
    else:
        raise ValueError(f"Unknown model type {arguments.get('model_type')}")
    logger.info(f"Model Type {arguments.get('model_type')} selected")
    summary(model, input_size=(arguments.get("batch_size"), 1, 28, 28))
    # Start Training  ⚡
    trainer = Trainer(max_epochs=arguments.get("epochs"))
    # External ModelSchemas
    # external_schemas = ExternalModelSchemas(input=input_schema, output=output_schema)
    # TrainingDataReference
    # tdataref = TrainingDataReference(schema=training_ref)
    trainer.fit(model)
    trainer.test()
    end = time.time()
    save_model(model=model, model_dir=arguments.get("model_dir"))
    trainer.save_checkpoint(
        filepath=os.path.join(arguments.get("model_dir"), "model_checkpoint.ckpt")
    )

    run_id = facts_client.runs.get_current_run_id()
    logger.info(f"Current RunID {run_id}")

    if os.environ.get("SM_TRAINING_ENV"):
        SM_TRAINING_ENV = json.loads(os.environ["SM_TRAINING_ENV"])
        logger.info(f"Log SM_TRAINING_ENV {os.environ['SM_TRAINING_ENV']}")
        facts_client.runs.set_tags(
            run_id,
            {
                "job_name": SM_TRAINING_ENV["job_name"],
                "user_entry_point": SM_TRAINING_ENV["user_entry_point"],
            },
        )
    else:
        logger.info("SM_TRAINING_ENV not in in environment variables")

    facts_client.runs.log_metrics(
        run_id, {"training_duration": np.round(end - start, 4)}
    )
    facts_client.runs.set_tags(run_id, {"git.branch": branch})

    experiments = facts_client.experiments.list_experiments()
    logger.info(f"experiments : {experiments}")

    runs = facts_client.runs.list_runs_by_experiment(EXPERIMENT_ID)
    logger.info(f"runs : {runs}")

    facts_client.export_facts.export_payload(run_id)

    fs_model = facts_client.external_model_facts.save_external_model_asset(
        model_identifier=EXPERIMENT_NAME,
        name=EXPERIMENT_NAME,
        catalog_id=args.get("catalog_id"),
        # schemas=external_schemas,
        # training_data_reference=tdataref,
        description="MNIST FC mdl trained Pytorch Lightning",
    )

    muc_utilities = facts_client.assets.get_ai_usecase(
        ai_usecase_id=arguments.get("model_entry_id"),
        catalog_id=arguments.get("catalog_id"),
    )

    fs_model.track(
        model_usecase=muc_utilities,
        approach=muc_utilities.get_approaches()[0],
        version_number="minor",  # "0.1.0"
    )


def save_model(model, model_dir):
    logger.info("Saving Pytorch lightning model")
    path = os.path.join(model_dir, "model.pth")
    weights = model.cpu().state_dict()
    torch.save(weights, path)
    return


if __name__ == "__main__":
    logger.info("Training MNIST model with Pytorch Lightning Framework")
    # Clean up MLFLOW and lighting_logs
    for dir in ["./mlruns", "./lighting_logs", "./logs"]:
        if os.path.isdir(dir):
            shutil.rmtree(dir)
    args = vars(parse_args())
    train(args)
