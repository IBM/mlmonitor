# SPDX-License-Identifier: Apache-2.0
import os
import json
import logging
from typing import Dict, Union
from mlmonitor.src import PROJECT_ROOT


class ModelPerturbator:
    """
    ModelPerturbator define a standard interface for Model Configuration agnostic of model serving environment
    """

    def __init__(self, source_dir: str, monitor_type: str, scenario_id: str):

        self.source_dir = source_dir
        self.monitor_type = monitor_type
        self.scenario_id = scenario_id

        self.load_perturbations(monitor_type, scenario_id)

    @property
    def source_dir(self):
        """source_dir Config property"""
        return self._source_dir

    @source_dir.setter
    def source_dir(self, value: str):
        """
        Set the source_dir to use to instantiate Config Class
        :param value:
        :return:
        """
        self._source_dir = value

    @property
    def total_records(self):
        """number of records"""
        return self._total_records

    @total_records.setter
    def total_records(self, value: int):
        """ """
        self._total_records = value

    @property
    def ratios(self):
        """the sequence of ratios used for applying the function"""
        return self._ratios

    @ratios.setter
    def ratios(self, value: list):
        """ """
        self._ratios = value

    @property
    def source_column(self):
        """source column on which the source condition is applied"""
        return self._source_column

    @source_column.setter
    def source_column(self, value: str):
        """ """
        self._source_column = value

    @property
    def source_cond(self):
        """condition"""
        return self._source_cond

    @source_cond.setter
    def source_cond(self, value: str):
        """ """
        self._source_cond = value

    @property
    def target_column(self):
        """target column on which the perturbation is applied"""
        return self._target_column

    @target_column.setter
    def target_column(self, value: str):
        """ """
        self._target_column = value

    @property
    def perturbation_fn(self):
        """function to apply"""
        return self._perturbation_fn

    @perturbation_fn.setter
    def perturbation_fn(self, value: str):
        """ """
        self._perturbation_fn = value

    def load_perturbations(
        self,
        monitor_type: str,
        scenario_id: str,
        file: Union[str, Dict] = None,
    ):
        """
        Load Model Perturbation from JSON file to ModelPerturbator Object
        :param file: json file containing a serialized version of a ModelConfig object
        :return:
        """
        if isinstance(file, str):
            with open(file, "r") as f:
                loaded_dict = json.load(f)
            logging.info(f"ModelPerturbator class loaded from file {file}")

        elif isinstance(file, dict):
            loaded_dict = file.copy()
            logging.info("ModelPerturbator class loaded from dict")
        else:
            file = f"{PROJECT_ROOT}/{self.source_dir}/model_perturbation.json"
            assert os.path.exists(
                file
            ), f"model_perturbation.json undefined for use case {self.source_dir}"

            with open(file, "r") as f:
                loaded_dict = json.load(f)
            logging.info("ModelPerturbator class loaded with default configuration")

        assert monitor_type in {
            "quality",
            "fairness",
            "explainability",
            "drift",
        }, "the monitor_type is invalid"

        assert (
            scenario_id in loaded_dict.get(monitor_type).keys()
        ), "the scenario_id does not exist"

        scenario = loaded_dict.get(monitor_type).get(scenario_id)

        self.total_records = scenario.get("total_records")
        self.ratios = scenario.get("ratios")
        self.perturbation_fn = scenario.get("perturbation_fn")
        self.target_column = scenario.get("target_column")

        if "source_column" in scenario.keys():
            self.source_column = scenario.get("source_column")
            self.source_cond = scenario.get("source_cond")

    def model_perturbation_json(self) -> Dict:
        """
        returns a dictionary containing the job_name, catalog_id, and model_endpoint of a SageMaker Model Use case.
        :param self: Access the class attributes
        :return: dictionary with job_name, catalog_id and model_endpoint as keys
        """

        if "_source_column" in self.__dict__.keys():
            return {
                "total_records": self.total_records,
                "ratios": self.ratios,
                "perturbation_fn": self.perturbation_fn,
                "source_column": self.source_column,
                "source_cond": self.source_cond,
                "target_column": self.target_column,
            }
        else:
            return {
                "total_records": self.total_records,
                "ratios": self.ratios,
                "perturbation_fn": self.perturbation_fn,
                "target_column": self.target_column,
            }


if __name__ == "__main__":
    from dotenv import load_dotenv
    import sys

    load_dotenv()
    sys.path.append(os.environ["PYTHONPATH"])

    from mlmonitor.src.demos.scenario_helpers import (
        perturb_column,
        perturb_double_column,
    )
    import pandas as pd
    import matplotlib.pyplot as plt

    # Loading scenario & args
    source_dir = "use_case_gcr"
    monitor_type = "drift"
    drift_type = "double"

    if drift_type == "single":
        scenario_id = "single_column_1"
    elif drift_type == "double":
        scenario_id = "double_column_1"

    perturbation_args = ModelPerturbator(
        source_dir=source_dir, monitor_type=monitor_type, scenario_id=scenario_id
    )
    perturbation_args = perturbation_args.model_perturbation_json()
    ratios_list = perturbation_args.get("ratios")

    # Reading the data
    valid_df = pd.read_csv(f"{PROJECT_ROOT}/datasets/gcr/test_feedback_data_gcr.csv")

    # Applying perturbation
    for ratio in ratios_list:
        print(f"Applying perturbation: {perturbation_args['perturbation_fn']}")

        if drift_type == "single":
            perturb_df = perturb_column(
                df=valid_df,
                total_records=perturbation_args["total_records"],
                ratio=ratio,
                target_column=perturbation_args["target_column"],
                perturbation_fn=perturbation_args["perturbation_fn"],
            )
        elif drift_type == "double":
            perturb_df = perturb_double_column(
                df=valid_df,
                total_records=perturbation_args["total_records"],
                ratio=ratio,
                source_column=perturbation_args["source_column"],
                source_cond=perturbation_args["source_cond"],
                target_column=perturbation_args["target_column"],
                perturbation_fn=perturbation_args["perturbation_fn"],
            )

        # Visual perturbations
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(valid_df["LoanAmount"].reset_index(drop=True), label="Original")
        ax1.plot(
            perturb_df["LoanAmount"].reset_index(drop=True),
            label="Perturbed",
            linestyle="--",
        )
        ax1.legend()
        ax1.set_title("LoanAmount")

        ax2.hist(
            valid_df["LoanAmount"].reset_index(drop=True),
            label="Original",
            alpha=0.5,
            bins=10,
        )
        ax2.hist(
            perturb_df["LoanAmount"].reset_index(drop=True),
            label="Perturbed",
            alpha=0.5,
            bins=10,
        )
        ax2.legend()
        ax2.set_title("LoanAmount Distribution")
        fig.tight_layout()
        plt.show()
