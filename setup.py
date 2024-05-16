# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List
from setuptools import setup
import os
import re


def version(path: str):
    with open(os.path.join(path, "version.meta"), "r") as v:
        return v.read().strip()


def readme(path: str):
    with open(os.path.join(path, "README.md"), "r") as v:
        return v.read()


def _readfile(file_path: str, file_name):
    with open(os.path.join(file_path, file_name), "r") as v:
        lines = v.readlines()
    return list(filter(lambda x: re.match(r"^\w+", x), lines))


def requirements(path: str, postfix: Optional[str] = None) -> List[str]:
    req = f"requirements-{postfix}.txt" if postfix else "requirements.txt"
    return _readfile(file_path=path, file_name=req)


current_directory = os.path.abspath(os.path.dirname(__file__))
MINIMUM_PYTHON_VERSION = "3.10.4"
DATA_FILES = ["*.csv", "*.gzip", "*.gz", "*.txt", "*.npy"]

setup(
    name="mlmonitor",
    version=version(current_directory),
    author="Jacques-Sylvain Lecointre",
    description="Orchestration of model use cases",
    author_email="js.lecointre@ibm.com",
    url="https://github.com/IBM/mlmonitor/mlmonitor",
    packages=[
        "mlmonitor",
        "mlmonitor.use_case_churn",
        "mlmonitor.use_case_gcr",
        "mlmonitor.use_case_mnist_tf",
        "mlmonitor.custmonitor",
        "mlmonitor.custmonitor.metrics",
        "mlmonitor.custmonitor.metricsprovider",
        "mlmonitor.data",
        "mlmonitor.figures",
        "mlmonitor.models",
        "mlmonitor.datasets",
        "mlmonitor.datasets.mnist",
        "mlmonitor.datasets.gcr",
        "mlmonitor.datasets.churn",
        "mlmonitor.src",
        "mlmonitor.src.utils",
        "mlmonitor.src.demos",
        "mlmonitor.src.model",
        "mlmonitor.src.azure",
        "mlmonitor.src.wml",
        "mlmonitor.src.aws",
        "mlmonitor.src.wos",
        "mlmonitor.src.factsheets",
    ],
    package_data={
        "mlmonitor.datasets": DATA_FILES,
        "mlmonitor.datasets.mnist": DATA_FILES,
        "mlmonitor.datasets.gcr": DATA_FILES,
        "mlmonitor.datasets.churn": DATA_FILES,
    },
    long_description=readme(path=os.path.join(current_directory, "mlmonitor")),
    long_description_content_type="text/markdown",
    install_requires=requirements(path=current_directory),
    extras_require={
        "local": requirements(path=current_directory, postfix="local"),
        "dev": requirements(path=current_directory, postfix="dev"),
        "azure": ["azureml-sdk==1.56.0"],
        "sagemaker": ["sagemaker==2.206.0"],
        "drift": ["ibm-wos-utils==4.7.0.14"],
    },
    python_requires=f">={MINIMUM_PYTHON_VERSION}",
    include_package_data=True,
)
