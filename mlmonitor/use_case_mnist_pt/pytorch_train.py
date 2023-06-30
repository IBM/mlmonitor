# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import sys
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ibm_aigov_facts_client import AIGovFactsClient, CloudPakforDataConfig

from pt_models import ConvNet
from utils import get_secret, parse_args
from torch_utils import MNIST

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Decode binary data from SM_CHANNEL_TRAINING
# Decode and preprocess data
# Create map dataset


def train(args):

    if os.environ.get("SM_TRAINING_ENV"):
        # verbose = 0
        region_name = args.get("region_name", "ca-central-1")
        SM_TRAINING_ENV = json.loads(os.environ["SM_TRAINING_ENV"])
        EXPERIMENT_NAME = SM_TRAINING_ENV["job_name"]
        logger.info(f"Running in a SM JOB {EXPERIMENT_NAME}")

        # Init Factsheet Client
        API_KEY_NAME = args.get("ibm_key_name")
        API_KEY = json.loads(
            get_secret(secret_name="IBM_KEYS", region_name=region_name)
        ).get(API_KEY_NAME)
        # install_dependencies()
    else:
        time_suffix = time.strftime("%Y%m%d-%H%M", time.gmtime())
        EXPERIMENT_NAME = f"sm-mnist-cnn-pytorch-{time_suffix}"
        API_KEY = os.getenv("API_KEY")

    start = time.time()

    if args.get("cp4d_env") == "saas":
        facts_client = AIGovFactsClient(
            api_key=API_KEY,
            experiment_name=EXPERIMENT_NAME,
            set_as_current_experiment=True,
            external_model=True,
            enable_autolog=False,
        )
    elif args.get("cp4d_env") == "prem":
        facts_client = AIGovFactsClient(
            cloud_pak_for_data_configs=CloudPakforDataConfig(
                service_url=args.get("cp4d_url"),
                username=args.get("cp4d_username"),
                api_key=API_KEY,
            ),
            experiment_name=EXPERIMENT_NAME,
            external_model=True,
            enable_autolog=False,
            set_as_current_experiment=True,
        )

    facts_client.manual_log.start_trace()

    # Train the model ⚡
    use_cuda = args.get("num_gpus") > 0
    device = torch.device("cuda" if use_cuda > 0 else "cpu")

    torch.manual_seed(args.get("seed"))
    if use_cuda:
        torch.cuda.manual_seed(args.get("seed"))

    train_loader = DataLoader(
        MNIST(args.get("train"), train=True),
        batch_size=args.get("batch_size"),
        shuffle=True,
    )
    test_loader = DataLoader(
        MNIST(args.get("test"), train=False),
        batch_size=args.get("test_batch_size"),
        shuffle=False,
    )

    net = ConvNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        betas=(args.get("beta_1"), args.get("beta_2")),
        weight_decay=args.get("weight_decay"),
    )

    logger.info("Start training ...")
    for epoch in range(1, args.get("epochs") + 1):
        net.train()
        for batch_idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            output = net(imgs)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.get("log_interval") == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(imgs),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        # test the model
        test(net, test_loader, device)

    # Log training time
    run_id = facts_client.runs.get_current_run_id()
    logger.info(f"Current RunID {run_id}")

    end = time.time()

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

    facts_client.runs.log_params(
        run_id=run_id,
        params={
            "epochs": args.get("epochs"),
            "batch_size": args.get("batch_size"),
            "learning_rate": args.get("batch_size"),
            "weight_decay": args.get("weight_decay"),
        },
    )

    facts_client.runs.log_metric(
        run_id=run_id, key="training_time", value=np.round(end - start, 4)
    )

    facts_client.manual_log.set_tags(
        tags={
            "engineering": "ML Platform",
            "release.candidate": "RC1",
            "release.version": "2.2.0",
        }
    )

    facts_client.export_facts.export_payload_manual(run_id)

    # Log external Model
    fs_model = facts_client.external_model_facts.save_external_model_asset(
        model_identifier=EXPERIMENT_NAME,
        name=EXPERIMENT_NAME,
        description="sagemaker Pytorch CNN MNIST",
    )
    muc_utilities = facts_client.assets.get_model_usecase(
        model_usecase_id=args.get("model_entry_id"),
        catalog_id=args.get("catalog_id"),
    )

    fs_model.track(
        model_usecase=muc_utilities,
        approach=muc_utilities.get_approaches()[0],
        version_number="minor",  # "0.1.0"
    )
    # save model checkpoint
    save_model(net, args.get("model_dir"))
    return


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            test_loss += F.cross_entropy(output, labels, reduction="sum").item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{}, {})\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return


def save_model(model, model_dir):
    logger.info("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    return


if __name__ == "__main__":
    # Clean up MLFLOW and lighting_logs
    for dir in ["./mlruns", "./lighting_logs"]:
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    args = vars(parse_args())
    train(args)
