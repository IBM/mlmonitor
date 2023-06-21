# SPDX-License-Identifier: Apache-2.0
from setuptools import setup
import os


def version(path: str):
    with open(os.path.join(path, "version.meta"), "r") as v:
        return v.read().strip()


def readme(path: str):
    with open(os.path.join(path, "README.md"), "r") as v:
        return v.read()


current_directory = os.path.abspath(os.path.dirname(__file__))

setup(
    name="custmonitor",
    version=version(current_directory),
    long_description=readme(path=os.path.join(current_directory, "custmonitor")),
    description="helpers for Watson OpenScale custom monitors",
    author="Jacques-Sylvain Lecointre",
    author_email="js.lecointre@ibm.com",
    url="https://github.com/IBM/mlmonitor/mlmonitor/custmonitor",
    packages=["custmonitor", "custmonitor.metricsprovider", "custmonitor.metrics"],
    include_package_data=True,
)
