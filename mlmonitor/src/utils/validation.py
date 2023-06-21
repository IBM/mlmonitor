# SPDX-License-Identifier: Apache-2.0
import os
import pandas as pd
import re

from mlmonitor.src import logger

# Regex: 32 characters separated by - following xxxxxxxx-xxxx-4xxx-xxxx-xxxxxxxxxxxx
_UUID4_REGEX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def is_directory(name: str) -> bool:
    """
    returns True if the given name is a directory, and False otherwise.

    :param name:str: name of the directory
    :return:bool: True if the name passed to it is a directory
    """
    return os.path.isdir(name)


def is_file(name: str) -> bool:
    """
    returns True if the given name is a file, and False otherwise.

    :param name:str: name of the file
    :return:bool: True if the name passed to it is a file
    """
    return os.path.isfile(name)


def is_csv(path: str) -> bool:
    """Ensure Path corresponds to csv file.

    path (str): string that should correspond to Path.
    RETURNS: boolean value is input path us a valid and readable csv file
    """
    if not is_file(path):
        return False
    if not path.endswith(".csv"):
        return False
    try:
        pd.read_csv(path, engine="python")
    except pd.errors.ParserError as e:
        logger.error(f"pandas.errors.ParserError {e}")
        return False
    return True


def exists(name):
    """
    returns True if the file exists, and False otherwise.

    :param name: Specify the file name
    :return: True if the file exists and false if it doesn't
    """
    return os.path.exists(name)


def validate_uuid4(value: str) -> bool:
    """
    checks if the value is a valid UUID4.

    :param value:str: Specify the value that you want to check
    :return: A boolean value
    #"""
    return bool(_UUID4_REGEX.match(value))


def validate_hdf_file(path: str) -> bool:
    extension_match = bool(re.compile(r"^/.*/.*.h5$").match(path))
    return extension_match and exists(path)
