# ML Model Lifecycle with mlmonitor

mlmonitor python module is designed to help you:

1. **Operate** a curated collection of model use cases, running on AWS Sagemaker,Azure and Watson Machine Learning, reporting metadata to AI Factsheets and monitored in IBM Watson OpenScale.

2. **Experience** an end-to-end model metadata capture from development to model operationalization states.

3. **Accelerate** the setup time of monitors such as quality, input and output drift, fairness, explainability and custom metrics.

4. **Run** drift and fairness demo scenarios to better understand algorithms supported by Watson OpenScale (e.g. [KS tests](https://www.statisticshowto.com/kolmogorov-smirnov-test/) , [bias mitigation algorithms](https://github.com/Trusted-AI/AIF360), [local post-hoc explanation](https://github.com/Trusted-AI/AIX360)).

## 0. Prerequisites

### 0.1 install mlmonitor package


```bash
$ pip install mlmonitor
```

This command will install **mlmonitor** with its dependencies such as  ***ibm-aigov-facts-client*** which is the AI factsheets client library built from this [repo](https://pypi.org/project/ibm-aigov-facts-client) .

Documentation is available [here](https://s3.us.cloud-object-storage.appdomain.cloud/factsheets-client/index.html)

**<u>Note :</u>** latest releases of this client can be installed from test PiPy server https://test.pypi.org/simple/git stat

### 0.2 Set `MONITOR_CONFIG_FILE` environment variable

populate [credentials.cfg](./credentials_example.cfg) file , see section 2 for more details .

Once this file is populated set `MONITOR_CONFIG_FILE` environment variable as follow :

```bash
$ export MONITOR_CONFIG_FILE=<GitHub-directory>/credentials.cfg
```

you are now setup to use `mlmonitor`

## 1. List of model use cases supported

|              Task              |              directory              |  Comment| train script  | inference script  |
| :--------------------------------------------- | :------------------------------------------------- |:---------- |:---------- |:---------- |
| Customer churn prediction (xgboost binary classification) | [use_case_churn](./use_case_churn) | <span style="color:green">*completed e2e*</span> |[train_cc_xg_boost.py](./use_case_churn/train_cc_xg_boost.py) | [inference_cc_xg_boost.py](./use_case_churn/inference_cc_xg_boost.py)|
| German Credit Risk (scikit binary classification) | [use_case_gcr](./use_case_gcr) | <span style="color:green">*completed e2e*</span> | [train_gcr.py](./use_case_gcr/train_gcr.py) | [inference_aws_gcr.py](./use_case_gcr/inference_aws_gcr.py)|
| Handwritten digit detection Keras |  [use_case_mnist_tf](./use_case_mnist_tf) | <span style="color:green">*completed e2e*</span> | [tf_cnn_train.py](./use_case_mnist_tf/tf_cnn_train.py) | [tf_cnn_inference.p](./use_case_mnist_tf/tf_cnn_inference.py)|
| Handwritten digit detection Pytorch  | [use_case_mnist_pt](./use_case_mnist_pt)  | <span style="color:orange">*train only*</span> | [pytorch_train.py](./use_case_mnist_pt/pytorch_train.py)  |[pytorch_inference.py](./use_case_mnist_pt/pytorch_inference.py)|
| Handwritten digit detection Pytorch Lightning | [use_case_mnist_ptlt](./use_case_mnist_ptlt) | <span style="color:orange">*train only*</span> | [ptlt_train.py](./use_case_mnist_ptlt/ptlt_train.py) |[ptlt_inference.py](./use_case_mnist_ptlt/ptlt_inference.py)|

## 2. configuration of `mlmonitor` lib details

### 2.1 Option 1 with configuration file

- Update [credentials.cfg](./credentials_example.cfg) file with your IBM API keys 🔑 , AWS credentials , COS details and OpenScale instance id...

- **saas** section is required For IBM Cloud environment

- **prem** section is required For Cloud Pak for Data running on OCP environment

  ```json
  {
    "saas": {
      "apikey": "xxxxx",
      "wml_url": "https://<cloud region>.ml.cloud.ibm.com",
      "wos_instance_id": "xxxxxxxxxxxxxxxxx",
      "default_space": "xxxxx",
      "cos_resource_crn" : "xxxxx",
      "cos_endpoint" : "https://s3.<cloud region>.cloud-object-storage.appdomain.cloud",
      "bucket_name" : "xxxxx"
  },
    "prem": {
      "version": "4.7",
      "username": "",
      "apikey": "xxxxx",
      "wos_instance_id" : "00000000-0000-0000-0000-000000000000",
      "wml_instance_id": "openshift",
      "default_space": "xxxxx",
      "ibm_auth_endpoint" : "xxxxxxxxxxxxxxxxx"
  },
    "aws_credentials" :
    {
      "access_key": "XXXXXXXXXXXXXXXXXXXXX",
      "secret_key": "XXXXXXXXXXXXXXXXXXXXX",
      "region_name": "XXXXXXXX"
    },
  "azure":
  {
    "client_id": "xxxxx",
    "client_secret": "xxxxx",
    "subscription_id": "xxxxx",
    "tenant_id": "xxxxx",
    "resource_group":"xxxxx",
    "workspace_name":"xxxxx"
  }
  }
  ```

- Set `ENV` environment variable to 'saas' or 'prem'
- Set `MONITOR_CONFIG_FILE` environment variable with the complete path for this configuration file

#### 2.1.1 SaaS environment ('saas' section)

**saas** section must be filled if services are running  on IBM Cloud with mandatory fields :

1. `apikey` : IBM Cloud API key to instantiate service instances for Watson Machine Learning and Watson OpenScale
2. `default_space` : deployment space to be used for WML models and custom metrics providers for custom monitors
3. `wml_url` : URL of Watson Machine Learning service (region specific)
4. `cos_resource_crn` : Cloud Resource Name for Cloud object storage to be used for storing  WML and WOS model traning data references.
5. `cos_endpoint` : Cloud object storage url (region specific) https://s3.us-east.cloud-object-storage.appdomain.cloud
6. `bucket_name` : Bucket name to be used for traning data reference uploads.

- `wos_instance_id` is optional but recommended to identify the SaaS service instance to be used.
- `wos_url` is optional and set by default to https://api.aiopenscale.cloud.ibm.com

#### 2.1.2 On prem environment ('prem' section)

**prem** section must be filled if services are running  on Cloud Pak for Data on OCP (CP4D >= 4.7 is required):

1. `apikey` : Cloud Pak for Data API key to instantiate service instances for Watson Machine Learning and Watson OpenScale
2. `default_space` : deployment space to be used for WML models and custom metrics providers for custom monitors
3. `version` : Cloud Pak for Data version
4. `username` : Cloud Pak for Data username to be used to clients
5. `ibm_auth_endpoint` : Cloud Pak for Data url


- `wos_instance_id` is optional but recommended to identify the Watson OpenScale instance defaut value is set to "00000000-0000-0000-0000-000000000000"
- `wml_instance_id` is optional default value is set to "openScale"

#### 2.1.3 Sagemaker credentials

**aws** section must be filled if models should be trained or deployed in Sagemaker

1. `access_key` : AWS access key with Sagemaker access
2. `secret_key` : AWS secret key with Sagemaker access
3. `region_name` : AWS region
4. `role` : Sagemaker execution role

### 2.2 Option 2 with environment variables

Alternatively you need to setup environment variables listed here.

- `API_KEY`
- `AUTH_ENDPOINT`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `AWS_ROLE`
- `COS_RESOURCE_CRN`
- `COS_ENDPOINT`
- `BUCKET_NAME`
- `WOS_URL`
- `WOS_INSTANCE_ID`
- `USERNAME`
- `VERSION`
- `WML_INSTANCE_ID`
- `WML_SPACE_ID`

## 3. How to use it

for more details , refer to Readme of each model use case e.g `use_case_churn`

```bash
(venv-mlmonitor)
$ python
>>> from mlmonitor import SageMakerModelUseCase
>>> source_dir = 'use_case_churn'
>>> catalog_id = <catalog id>
>>> model_entry_id = <Model use case id for churn models>
>>> model_use_case = SageMakerModelUseCase(source_dir,catalog_id,model_entry_id)
>>> model_use_case.train()

>>> model_use_case.display_states()

              - trained:   [True]
              - deployed:  [False]
              - governed:  [False]
              - monitored: [False]
```


## 4. Onboard a new ML model using mlmonitor

Each model use case should be placed under a folder following this naming convention `use_case_<use case name>`.

- Please refer to documentation of each model use case to deploy and monitor a model using *mlmonitor*
- German Credit Risk [Readme](./use_case_gcr/README.md) has detailed instructions.

### 4.1 AWS code repositories

They should contain :

    ├── use_case_<use case name>               <- ALL THIS CODE DIRECTORY is SHIPPED AND EXECUTED IN AWS SAGEMAKER
    │   ├── __init__.py    <- Makes use_case a Python module
    │   ├── train.py       <- Train script to be executed in SM
    │   ├── inference.py   <-  Inference script to be executed in SM
    │   ├── test_inference.py <- test inference scripts on locally trained model before deploying to AWS
    │   ├── test_train.py <- test training scripts before sending training job to AWS Sagemaker
    │   └── model_signature.json <- all the model details to onboard this model in WOS and AWS SM
    │   └── requirements.txt <- all these dependencies (included Factsheets client) to be installed on AWS container

These directories are shipped to AWS at training or deployment time and should contain custom training script(s) , custom inference script(s) and dependencies.

### 4.1 model signature files

Each model use case should be self-contained and include ***model_signature.json*** file following the structure

```json
{
"signature": {
  "feature_columns": [],
  "class_label": "",
  "prediction_field": "",
  "probability_fields": [],
  "categorical_columns": [],
  "problem_type": "",
  "data_type": "structured",
  "description": "description"
},

"datasets": {
"training_data": "",
"validation_data": "",
"test_data": ""
},
"aws_runtime": {
    "train_script": "train_cc_xg_boost.py",
    "inference_script": "inference_cc_sk.py",
    "train_framework": "xgboost",
    "train_framework_version": "1.5-1",
    "train_py_version": "py3",
    "inference_framework": "sklearn",
    "inference_framework_version": "1.0-1",
    "train_instance": "ml.m4.xlarge",
    "inference_instance": "ml.m4.xlarge",
    "inference_py_version": "py3",
    "prefix": "sagemaker/DEMO-xgboost-churn",
    "job_name": "sm-cc-xgboost",
    "serializer": "json",
    "deserializer": "json"
},
"wml_runtime": {
    "train_module": "train_cc_xg_boost",
    "train_method": "train_wml",
    "inference_instance": "runtime-22.2-py3.10",
    "inference_framework": "scikit-learn",
    "inference_framework_version": "1.1",
    "inference_script": "inference_cc_sk.py"
},
"azure_runtime": {
    "train_script": "train_cc_xg_boost.py",
    "train_py_version": "3.8",
    "inference_script": "inference_cc_sk.py",
    "inference_compute": "aci",
    "aks_cluster_name": "aks-cluster",
    "cpu_cores": 1,
    "memory_gb": 1,
    "auth_enabled": false,
    "description" : "Customer Churn prediction - monitored in WOS",
    "tags" : {"data": "customer churn", "method": "xgboost"},
    "conda_packages": ["pandas==1.5.2", "boto3","seaborn", "matplotlib"],
    "pip_packages": ["ibm-aigov-facts-client==1.0.59","xgboost==1.6.1","scikit-learn==1.0.1","ibm_watson_openscale==3.0.27" ,"pygit2"],
    "inference_py_version": "3.9",
    "train_module": "train_cc_xg_boost",
    "train_method": "train_wml"
},
"hyperparameters" : {
    "max_depth": 5,
    "eta": 0.2,
    "gamma": 4,
    "min_child_weight": 6,
    "subsample": 0.8,
    "objective": "binary:logistic",
    "num_round": 200,
    "verbosity": 0
},

"quality_monitor" : {
    "enabled": true,
    "parameters": {"min_feedback_data_size": 10},
    "thresholds": [
        {"metric_id": "area_under_roc", "type": "lower_limit", "value": 0.80}
    ]
},

"fairness_monitor" : {
    "enabled": true,
    "parameters": {
        "features": [
            {
                "feature": "Day Mins",
                "majority": [[2.501, 5.330], [5.331, 7.936], [7.937, 20]],
                "minority": [[0.000, 2.500]],
                "threshold": 0.95
            }
        ],
        "favourable_class": [0],
        "unfavourable_class": [1],
        "min_records": 100
    }
},

"drift_monitor" : {
    "enabled": true,
    "parameters": {
        "min_samples": 100,
        "drift_threshold": 0.1,
        "train_drift_model": false,
        "enable_model_drift": true,
        "enable_data_drift": true
    },
    "learn_constraints": {
        "two_column_learner_limit": 200,
        "categorical_unique_threshold": 0.8,
        "user_overrides": []
    }
},

"explain_monitor" : {"enabled": true}

"mrm_monitor" : {"enabled": true}

"custom_monitor" : {
    "enabled": true,
    "names": ["tp", "fp", "tn", "fn", "cost", "total"],
    "thresholds": [200, 10, 200, 10, 6000, 200],
    "provider_name": "Custom_Metrics_Provider_churn",
    "custom_monitor_name": "Custom_Metrics_Provider_churn",
    "wml_function_provider": "Custom_Metrics_Provider_Deployment_churn-deploy"}
}
```

### 4.2 model perturbation files

Similarly to ***model_signature.json***, each model use case should include ***model_perturbation.json*** file following the structure

```json
{
    "drift": {
        "single_column_1": {
            "total_records": 100,
            "ratios": [0.01, 0.05, 0.1, 0.2, 0.3],
            "target_column": "LoanAmount",
            "perturbation_fn": "x + 15000"
        },
        "single_column_2": {
            "total_records": 100,
            "ratios": [0.1, 0.2, 0.4, 0.8, 1.0],
            "target_column": "LoanAmount",
            "perturbation_fn": "x + 15000"
        },
        "double_column_1": {
            "total_records": 100,
            "ratios": [0.1, 0.2, 0.3, 0.6],
            "source_column": "LoanPurpose",
            "source_cond": "car_used",
            "target_column": "LoanAmount",
            "perturbation_fn": "x + np.mean(x)*100"
        }
    }
}
```

The JSON file uses the following nomenclature:

```
{
    <monitor type>: {
        <scenario ID>: {
            <scenario parameters>: <parameters values>
        }
    }
}
```

In each **scenario_id**, the following parameters can be used:

* `total_records`: number of records sent to Watson OpenScale in each iteration
* `ratios`: a list of percentages used to iterate over. In each iteration, the percentage defines the ratio of records to perturb
* `target_column`: column where to apply the perturbation
* `perturbation_fn`: perturbation function applied on the target column
* `source_column`: for two-column constraints, column used to filter the data
* `source_condition`: for two-column constraints, filter condition for the source column

Currently, there are two ways to apply drift on the payload data: single column perturbation or two-column perturbation. If `source_column` and `source_cond` are defined in the scenario, two-column perturbation will be applied.

These parameters are then used by the `ModelPerturbator` object to perturb the payload data sent to Watson OpenScale.
