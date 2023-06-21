# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import tarfile
import gzip
import os
import re


def load_json_from_file(file_path):
    """
    loads a JSON file from the specified path and returns it as a dictionary.

    :param file_path: Specify the file that is to be loaded
    :return: A dictionary
    """
    with open(file_path) as json_file:
        return json.load(json_file)


def make_model_tgzfile(output_filename: str, source_dir: str, filename: str) -> str:
    """
    creates a gzipped tar file containing the model file in the specified directory.
    The output filename is given by the argument 'output_filename'.
    The source directory is given by 'source_dir' and should be an absolute path.


    :param output_filename:str: the name of the tgz file that you want to create
    :param source_dir:str:  directory where the model is stored
    :param filename:str: name of the file to be compressed
    :return:
    """
    temp_unzipped_dir = tempfile.mktemp()
    current_dir = os.path.abspath(os.getcwd())
    assert os.path.exists(source_dir)
    os.chdir(source_dir)
    try:
        with tarfile.open(temp_unzipped_dir, "w") as tar:
            tar.add(filename)

        with gzip.GzipFile(
            filename="", fileobj=open(output_filename, "wb"), mode="wb", mtime=0
        ) as gzipped_tar, open(temp_unzipped_dir, "rb") as tar:
            gzipped_tar.write(tar.read())
    finally:
        os.remove(temp_unzipped_dir)
    os.chdir(current_dir)
    return os.path.join(source_dir, output_filename)


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


def requirements(path: str):
    return _readfile(file_path=path, file_name="requirements.txt")
