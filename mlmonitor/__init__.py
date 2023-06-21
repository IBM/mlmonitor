# SPDX-License-Identifier: Apache-2.0
"""The ``mlmonitor`` module provides a high-level API for enabling AI Governance for model use cases
running on a variety of Model serving platforms such as Watson Machine Learning , Sagemaker , Azure ML. Each Model is
governed in AI FatcSheets and Monitored in IBM Watson OpenScale.

Onboarding a new model can be achieved by:

.. code:: python

    from mlmonitor import SageMakerModelUseCase

    model_use_case = SageMakerModelUseCase(source_dir='use_case_churn',catalog_id=catalog_id,model_entry_id=model_entry_id)
    model_use_case.train()
    model_use_case.deploy()
    model_use_case.display_states()
    model_use_case.govern()
    model_use_case.monitor()
    model_use_case.configure_quality_monitor()
    model_use_case.configure_fairness_monitor()
    model_use_case.configure_explainability_monitor()
    model_use_case.configure_drift_monitor()
    model_use_case.log_payload()
    model_use_case.log_feedback()
    model_use_case.evaluate_model()
    model_use_case.save_use_case()

"""
from mlmonitor.src.model.use_case_wml import WMLModelUseCase  # noqa: F401

supported_uc = ["WMLModelUseCase"]

try:
    import sagemaker

    print(f"sagemaker installed at {sagemaker.__path__}")
    from mlmonitor.src.model.use_case_aws import SageMakerModelUseCase  # noqa: F401

    supported_uc.append("SageMakerModelUseCase")
except ModuleNotFoundError:
    print("run pip install mlmonitor[sagemaker] to use SageMakerModelUseCase")

try:
    import azureml

    print(f"azureml installed at {azureml.__path__}")
    from mlmonitor.src.model.use_case_azure import AzureModelUseCase  # noqa: F401

    supported_uc.append("AzureModelUseCase")
except ModuleNotFoundError:
    print("run pip install mlmonitor[azure] to use AzureModelUseCase")

__all__ = supported_uc
__author__ = "IBM Client Engineering."

try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("mlmonitor").version
except Exception:
    __version__ = "N/A"
