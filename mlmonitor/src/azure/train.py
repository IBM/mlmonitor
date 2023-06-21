# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional, List, Set
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import (
    Dataset,
    Model,
    Workspace,
    Datastore,
    Environment,
    Experiment,
    ScriptRunConfig,
)
from azureml.core.conda_dependencies import CondaDependencies

from mlmonitor.src.model.config_azure import AzureModelConfig
from mlmonitor.src import PROJECT_ROOT, API_KEY


def create_experiment(workspace: Workspace, experiment_name: str) -> Experiment:
    """
    creates an experiment in the workspace.

    :param workspace: Workspace: Azure workspace
    :param experiment_name: str: Name the experiment
    :return: Experiment object
    """
    exp = Experiment(workspace=workspace, name=experiment_name)
    return exp


def register_env(
    workspace: Workspace,
    name: str,
    python_version: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    conda_packages: Optional[List] = None,
    pip_packages: Optional[List] = None,
) -> Environment:
    """
    creates an environment in the Azure Machine Learning workspace for training jobs

    :param workspace: Workspace: Azure workspace
    :param name: str: Name the environment
    :param python_version: str: version of python to use in the environment
    :param api_key: Optional[str]: Pass the api key for IBM Cloud pak for Data
    :param model_name: Optional[str]: name of the model to register at training job completion
    :param conda_packages: Optional[List]: conda packages to be installed in the environment
    :param pip_packages: Optional[List]: pip packages to install in the environment
    :return: An environment object
    """
    env = Environment(name)

    if not pip_packages:
        pip_packages = []

    if not conda_packages:
        conda_packages = []

    conda_packages.extend(("pip", "azure-keyvault-secrets", "azure-identity"))

    pip_packages.extend(("ipython==8.12.0",))

    conda_deps = CondaDependencies.create(
        pip_packages=pip_packages, conda_packages=conda_packages
    )
    conda_deps.set_python_version(python_version)
    env.python.conda_dependencies = conda_deps
    if api_key:
        env.environment_variables |= {"API_KEY": api_key}
    if model_name:
        env.environment_variables |= {"JOB_NAME": model_name}

    env.register(workspace=workspace)
    return env


def create_aml_target(
    workspace: Workspace,
    compute_name: Optional[str] = None,
    compute_min_nodes: Optional[int] = None,
    compute_max_nodes: Optional[int] = None,
    vm_size: Optional[str] = None,
) -> ComputeTarget:
    """
    creates a new Azure Machine Learning compute target, or returns an existing one with compute_name

    :param workspace: Workspace: Azure workspace
    :param compute_name: Optional[str]: compute target name to use or create
    :param compute_min_nodes: Optional[int]: minimum number of nodes in the compute
    :param compute_max_nodes: Optional[int]: maximum number of nodes in compute
    :param vm_size: Optional[str]: type of vm to use for training job
    :return: An AmlCompute object
    """
    compute_name = compute_name or os.environ.get(
        "AML_COMPUTE_CLUSTER_NAME", "cpu-cluster"
    )
    compute_min_nodes = compute_min_nodes or os.environ.get(
        "AML_COMPUTE_CLUSTER_MIN_NODES", 0
    )
    compute_max_nodes = compute_max_nodes or os.environ.get(
        "AML_COMPUTE_CLUSTER_MAX_NODES", 4
    )
    # CPU VM by default
    vm_size = vm_size or os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

    if compute_name in workspace.compute_targets:
        compute_target = workspace.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print(f"found compute target: {compute_name}")
    else:
        print("creating new AmlCompute compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, min_nodes=compute_min_nodes, max_nodes=compute_max_nodes
        )

        # create the cluster
        compute_target = ComputeTarget.create(
            workspace=workspace,
            name=compute_name,
            provisioning_configuration=provisioning_config,
        )
        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20
        )

        # For a more detailed view of current AmlCompute status, use get_status()
        print(compute_target.get_status().serialize())

    return compute_target


def upload_to_datastore(
    workspace: Workspace,
    target_data_path: str,
    datafiles: Set,
    datastore_name: Optional[str] = None,
):
    """
    uploads a required files for model training to datastore.

    :param workspace: Workspace: Azure workspace
    :param target_data_path: str: path to which files should be uploaded
    :param datafiles: Set: files to upload
    :param datastore_name: Optional[str]: name of the datastore to use , default one used if not specified
    :return: None
    """
    datastore = (
        Datastore.register_azure_blob_container(
            workspace=workspace,
            datastore_name=datastore_name,
            container_name="",  # TODO add container_name
            account_name="",  # TODO add account_name
            account_key="",
        )
        if datastore_name
        else workspace.get_default_datastore()
    )
    datastore.upload_files(
        files=list(datafiles),
        target_path=target_data_path,
        overwrite=True,
        show_progress=True,
    )

    print(
        f"uploading  {len(datafiles)} File(s)to Datastore  for workspace {workspace.name} : [{datastore.name}]"
    )


def register_dataset(
    workspace: Workspace, datastore: Datastore, data_path: str
) -> Dataset:
    """
    Register a Dataset object from the files in data_path directory on the datastore.

    :param workspace: Workspace: Azure workspace
    :param datastore: Datastore: Azure Datastore object to use
    :param data_path: str: path to the data in datastore
    :return: Azure Dataset object
    """
    file_dataset = Dataset.File.from_files(path=(datastore, data_path))

    for fp in file_dataset.to_path():
        print(fp)

    return file_dataset.register(
        workspace=workspace,
        name=data_path,
        description=f"mlmonitor train {data_path}",
        create_new_version=True
        # tags={},
    )


def train_az_ml_job(
    catalog_id: str,
    model_entry_id: str,
    model_name: str,
    model_config: AzureModelConfig,
    workspace: Workspace,
    data_path: str,
    ibm_key_name: str = "IBM_API_KEY_MLOPS",
    cp4d_env: str = "saas",
    cp4d_username: Optional[str] = None,
    cp4d_url: Optional[str] = None,
) -> Model:
    """
    rain a model using Azure ML Job.

    :param model_name: str: Name of the model to register in Azure
    :param model_config: AzureModelConfig: model configuration object
    :param workspace: Azure workspace
    :param model_entry_id: str: model use case identifier in Cloud Pak for Data where to register model
    :param catalog_id: str: catalog identifier for the model use case in Cloud Pak for Data  where to register model
    :param data_path: str: Specify the location of the data files used by training a job
    :param ibm_key_name: str: Specify the name of the CP4D api key to be used for training (stored in keyvault)
    :param cp4d_env: str: type of CP4D environment - should be 'prem' or 'saas'
    :param cp4d_username: Optional[str]: username in CP4D  - required only for 'prem' cp4d_env
    :param cp4d_url: Optional[str]: CP4D url - required only for 'prem' cp4d_env
    :return: An Azure model object
    """
    env_name = f"mlmonitor-{model_config.source_dir}"
    compute_target = create_aml_target(workspace=workspace)

    exp = create_experiment(workspace=workspace, experiment_name=env_name)

    env = register_env(
        name=env_name,
        workspace=workspace,
        python_version=model_config.train_py_version,
        conda_packages=model_config.conda_packages,
        pip_packages=model_config.pip_packages,
        api_key=API_KEY,
        model_name=model_name,
    )

    datafiles = [
        model_config._get_data_location(dataset_type="train", base_data_dir=data_path),
        model_config._get_data_location(
            dataset_type="validation", base_data_dir=data_path
        ),
        model_config._get_data_location(dataset_type="test", base_data_dir=data_path),
    ]
    print(f"DATAFILE :\n{datafiles}")
    # datafiles = [
    #     os.path.join(data_path, model_config.data_dir, model_config.training_data),
    #     os.path.join(data_path, model_config.data_dir, model_config.validation_data),
    #     os.path.join(data_path, model_config.data_dir, model_config.test_data),
    # ]

    upload_to_datastore(
        workspace=workspace,
        target_data_path=model_config.data_dir,
        datafiles=set(datafiles),
    )

    entry_point = model_config.train_script
    source_dir = os.path.join(PROJECT_ROOT, model_config.source_dir)
    data_store = workspace.get_default_datastore()

    # register dataset
    dataset = register_dataset(
        workspace=workspace, datastore=data_store, data_path=model_config.data_dir
    )

    # Mandatory script parameters for each model use case
    # CP4D specific arguments
    # Training Job specific arguments
    args = [
        "--catalog-id",
        catalog_id,
        "--model-entry-id",
        model_entry_id,
        "--ibm-key-name",
        ibm_key_name,
        "--cp4d-env",
        cp4d_env,
        "--model-dir",  # output to store model artifacts
        "./outputs",
        "--train",  # directory for training data
        dataset.as_mount(),
        "--test",  # directory for test data
        dataset.as_mount(),
        "--validation",  # directory for validation data
        dataset.as_mount(),
        "--model-name",
        model_name,
    ]

    if cp4d_username:
        args.append("--cp4d-username")
        args.append(cp4d_username)
    if cp4d_url:
        args.append("--cp4d-url")
        args.append(cp4d_url)
    # Model specific hyperparameters
    if model_config.hyperparameters:
        for k, v in model_config.hyperparameters.items():
            args.extend((f"--{k}", v))

    run_config = ScriptRunConfig(
        source_directory=source_dir,
        script=entry_point,
        arguments=args,
        compute_target=compute_target,
        environment=env,
    )

    run = exp.submit(config=run_config)
    run.wait_for_completion(show_output=True)

    # TODO get model file(s) for different frameworks
    outputs = [
        file_name
        for file_name in run.get_file_names()
        if file_name.split("/")[0] == "outputs"
    ]

    if len(outputs) == 1:
        model_data = outputs[0]
    else:
        raise ValueError(f"multiple outputs produced in run:\n{outputs}")

    model = run.register_model(model_name=model_name, model_path=model_data)

    return model
