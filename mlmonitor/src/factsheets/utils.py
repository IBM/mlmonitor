# SPDX-License-Identifier: Apache-2.0
import requests
import json
from enum import Enum
from typing import Dict, Optional


class MLData(Enum):
    TRAIN = "Training"
    TEST = "Test"
    VALIDATE = "Validation"


class FactsheetHelpers:
    def __init__(
        self,
        api_key,
        container_type,
        container_id,
        model_entry_id,
        env: str = "saas",
        cpd_url: Optional[str] = None,
        username: Optional[str] = None,
    ):
        assert env in {"saas", "prem"}, "Environment should be set to saas or prem"
        if env == "prem":
            assert username, "Username should be set for CP4D on Prem environment"
            assert cpd_url, "CP4D url should be set for CP4D on Prem environment"

        self.env = env
        self.cpd_url = cpd_url
        self.base_url = (
            "https://api.dataplatform.cloud.ibm.com" if env == "saas" else self.cpd_url
        )
        self.api_key = api_key
        self.username = username
        self.container_type = container_type
        self.container_id = container_id
        self.model_entry_id = model_entry_id

    def _authenticate_prem(self, verify: bool = False) -> requests.Response:
        """
        uses the apikey to get an access token from CP4D running on OCP.
        It returns a JSON object with the access token and expiration date.

        :param url: CP4D base url
        :param username: CP4D username
        :param apikey:  CP4D API key
        :return: The access token for the IAM service
        """
        auth_url = f"{self.cpd_url}/icp4d-api/v1/authorize"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {"username": self.username, "api_key": self.api_key}
        token_response = requests.post(
            auth_url, headers=headers, json=payload, verify=verify
        )

        return token_response

    def _authenticate_saas(self, verify: bool = True) -> requests.Response:
        """Calls the authentication endpoint for Cloud Pak for Data as a Service,
        and returns authentication headers if successful.
        See https://cloud.ibm.com/apidocs/watson-data-api#creating-an-iam-bearer-token.
        Note this function is not cached by Streamlit since they token eventually expires, so users
        need to re-authenticate periodically.
        Args:

        Returns:
            success (bool): Whether authentication was successful
            headers (dict): If success=True, a dictionary with valid authentication headers. Otherwise, None.
            error_msg (str): The text response from the authentication request if the request failed.
        """

        data = {
            "apikey": self.api_key,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        }

        token_response = requests.post(
            url="https://iam.cloud.ibm.com/identity/token", data=data, verify=verify
        )

        return token_response

    def authenticate(self, verify: bool = True):
        if self.env == "saas":
            token_response = self._authenticate_saas(verify=verify)
            tkn_key = "access_token"
        else:
            token_response = self._authenticate_prem(verify=verify)
            tkn_key = "token"
        if token_response.ok:
            headers = {
                "Authorization": f"Bearer {token_response.json()[tkn_key]}",
                "content-type": "application/json",
            }
            return True, headers, ""
        else:
            print(token_response.text)
            return False, None, token_response.text

    def define_custom_model_facts(self, csv_file_name):
        from ibm_aigov_facts_client import FactSheetElements

        facts_elements = FactSheetElements(self.api_key)
        facts_elements.replace_asset_properties(
            csv_file_name, type_name="modelfacts_user", overwrite=True
        )

    def define_custom_model_entry_facts(self, csv_file_name):
        from ibm_aigov_facts_client import FactSheetElements

        facts_elements = FactSheetElements(self.api_key)
        facts_elements.replace_asset_properties(
            csv_file_name, type_name="model_entry_user", overwrite=True
        )

    def reset_model_entry_user(self, verify: bool = True):
        response = self.authenticate()
        auth_headers = response[1]
        api_url = f"{self.base_url}/v2/assets/{self.model_entry_id}/attributes?catalog_id={self.container_id}"
        body = {"name": "model_entry_user", "entity": {}}
        response = requests.post(
            api_url, data=json.dumps(body), headers=auth_headers, verify=verify
        )
        print(response.json())
        print(response.status_code)
        return response.status_code

    # Internal helper to set the value for a custom Model use case fact
    def _set_fact(
        self,
        fact_ids,
        fact_values,
        fact_type,
        asset_id,
        is_array=None,
        op="add",
        verify: bool = True,
    ):
        response = self.authenticate(verify=verify)
        auth_headers = response[1]
        api_url = f"{self.base_url}/v2/assets/{asset_id}/attributes/{fact_type}?{self.container_type}_id={self.container_id}"
        if is_array:
            body = [
                {"op": op, "path": f"/{fact_id}", "value": fact_value.split("|")}
                if arr
                else {"op": op, "path": f"/{fact_id}", "value": fact_value}
                for fact_id, fact_value, arr in zip(fact_ids, fact_values, is_array)
            ]
        else:
            body = [
                {"op": op, "path": f"/{fact_id}", "value": fact_value}
                for fact_id, fact_value in zip(fact_ids, fact_values)
            ]

        response = requests.patch(
            api_url, data=json.dumps(body), headers=auth_headers, verify=verify
        )
        print(response.text)
        return response.status_code

    def set_model_entry_fact(
        self, fact_ids, fact_values, is_array=None, op="add", verify: bool = True
    ):
        return self._set_fact(
            fact_ids=fact_ids,
            fact_values=fact_values,
            asset_id=self.model_entry_id,
            fact_type="model_entry_user",
            is_array=is_array,
            op=op,
            verify=verify,
        )

    def set_custom_fact(
        self,
        fact_ids,
        fact_values,
        model_id,
        is_array=None,
        op="add",
        verify: bool = True,
    ):
        return self._set_fact(
            fact_ids=fact_ids,
            fact_values=fact_values,
            asset_id=model_id,
            fact_type="modelfacts_user",
            is_array=is_array,
            op=op,
            verify=verify,
        )

    def get_model_entry_content(self, verify: bool = True) -> Dict:
        response = self.authenticate(verify=verify)
        auth_headers = response[1]
        api_url = f"{self.base_url}/v2/assets/{self.model_entry_id}/attributes?{self.container_type}_id={self.container_id}&allow_metadata_on_dpr_deny=true"
        response = requests.get(api_url, headers=auth_headers, verify=verify)
        return response.json()

    def get_models(self, verify: bool = True) -> Dict:
        response = self.authenticate(verify=verify)
        auth_headers = response[1]
        api_url = f"{self.base_url}/v1/aigov/model_inventory/model_entries/{self.model_entry_id}/models?{self.container_type}_id={self.container_id}&allow_metadata_on_dpr_deny=true"
        print(api_url)
        response = requests.get(api_url, headers=auth_headers, verify=verify)
        return response.json()

    def get_model_entry(self, verify: bool = True) -> Dict:
        response = self.authenticate(verify=verify)
        auth_headers = response[1]
        api_url = f"{self.base_url}/v1/aigov/model_inventory/model_entries/{self.model_entry_id}?{self.container_type}_id={self.container_id}"
        response = requests.get(api_url, headers=auth_headers, verify=verify)
        return response.json()

    def relatemodels(
        self, reference_model_id: str, model_id: str, verify: bool = True
    ) -> Dict:
        response = self.authenticate(verify=verify)
        auth_headers = response[1]
        api_url = f"{self.base_url}/v1/aigov/model_inventory/model_entries/{self.model_entry_id}/relatemodels?{self.container_type}_id={self.container_id}&allow_metadata_on_dpr_deny=true"
        response = requests.patch(
            api_url,
            headers=auth_headers,
            data=json.dumps(
                {"reference_model_id": reference_model_id, "model_id": model_id}
            ),
            verify=verify,
        )
        return response.json()

    def unlink_model_asset_from_entry(
        self,
        model_asset_id: str,
        container_id,
        container_type: str,
        verify: bool = True,
    ):
        response = self.authenticate(verify=verify)
        auth_headers = response[1]
        api_url = f"{self.base_url}/v1/aigov/model_inventory/models/{model_asset_id}/model_entry?{container_type}_id={container_id}"
        print(api_url)
        response = requests.delete(api_url, headers=auth_headers, verify=verify)
        return response.status_code

    def delete_asset(
        self,
        model_asset_id: str,
        container_id,
        container_type: str,
        verify: bool = True,
    ):
        response = self.authenticate(verify=verify)
        auth_headers = response[1]
        api_url = f"{self.base_url}/v2/assets/{model_asset_id}?{container_type}_id={container_id}"
        print(api_url)
        response = requests.delete(api_url, headers=auth_headers, verify=verify)
        return response.status_code

    # https://api.dataplatform.dev.cloud.ibm.com/v1/aigov/factsheet/api/explorer/#/WKC_Factsheet_Model_Entry_API
    def get_factsheet_asset_types(self, verify: bool = True):
        response = self.authenticate(verify=verify)
        return response.json()


def get_model_id_by_deployment_name(
    models: dict, deployment_name: str, key: Optional[str] = "id"
) -> Optional[str]:
    filtered_models = [
        physical_model[key] if key else physical_model
        for physical_model in models.get("physical_models")
        if deployment_name in [dep.get("name") for dep in physical_model["deployments"]]
    ]
    if len(filtered_models) == 1:
        return filtered_models[0]
    elif len(filtered_models) == 0:
        return None
    else:
        raise ValueError(
            f"Number of  models found for deployment_name  {deployment_name} !=1 => {len(filtered_models)} "
        )


def get_model_id_by_model_name(
    models: dict, model_name: str, state: str = "development", key: Optional[str] = "id"
) -> Optional[str]:
    supported_states = ["development", "pre-production", "production"]
    if state not in supported_states:
        raise ValueError(
            f"model state must be in {supported_states} => {state} passed "
        )

    filtered_models = [
        model[key] if key else model
        for model in models.get("physical_models")
        if model["name"] == model_name and model["deployment_space_type"] == state
    ]
    if len(filtered_models) == 1:
        return filtered_models[0]
    elif len(filtered_models) == 0:
        return None
    else:
        raise ValueError(
            f"Number of  models found for model_name  {model_name} !=1 => {len(filtered_models)} "
        )
