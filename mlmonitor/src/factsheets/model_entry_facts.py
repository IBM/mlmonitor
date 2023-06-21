# SPDX-License-Identifier: Apache-2.0
import os
import json
import pandas as pd

from mlmonitor.src import USERNAME, AUTH_ENDPOINT, API_KEY
from mlmonitor.src import PROJECT_ROOT, VERIFY_CP4D_SSL
from mlmonitor.src.factsheets.utils import FactsheetHelpers


def populate_model_entry_facts(
    model_entry_id: str,
    catalog_id: str,
    use_case: str,
    apikey: str = API_KEY,
):
    """
    Creates a model asset schema and populate default values using WKC API
    Parameters
    ----------
    model_entry_id : str
       IBM AI Factsheets Model use case identifier to be used to register the deployed model
    apikey : str
       IBM Cloud API key to use for authentication
    catalog_id : str
       IBM AI Factsheets catalog identifier to be used to register the deployed model
    use_case : str
       use case name
    Returns
    -------
    model_entry_details : dict
    dictionary with Model use case details
    {
    }
    """

    modelentry_user = os.path.join(
        f"{PROJECT_ROOT}/factsheets_{use_case}/",
        f"custom_model_entry_facts_{use_case}.csv",
    )

    modelentry_user_df = pd.read_csv(modelentry_user)
    df_modelentryfacts_values = pd.read_csv(
        f"{PROJECT_ROOT}/factsheets_{use_case}"
        f"/custom_model_entry_facts_values_{use_case}.csv",
        sep=";",
    ).loc[:, ["name", "default_value"]]

    fs_helper = FactsheetHelpers(
        api_key=apikey,
        container_type="catalog",
        container_id=catalog_id,
        model_entry_id=model_entry_id,
        username=USERNAME,
        cpd_url=AUTH_ENDPOINT,
    )

    df_modelentryfacts = df_modelentryfacts_values.merge(
        modelentry_user_df.loc[:, ["name", "type", "is_array"]], on="name", how="left"
    )
    fs_helper.define_custom_model_entry_facts(modelentry_user)

    # fs_helper.reset_model_entry_user()
    # print(df_["default_value"].to_list())

    fs_helper.set_model_entry_fact(
        fact_ids=df_modelentryfacts["name"].to_list(),
        fact_values=df_modelentryfacts["default_value"].to_list(),
        is_array=df_modelentryfacts["is_array"].to_list(),
        op="add",
        verify=VERIFY_CP4D_SSL,
    )

    model_entry_details = fs_helper.get_model_entry_content(verify=VERIFY_CP4D_SSL)
    with open(
        os.path.join(
            f"{PROJECT_ROOT}/factsheets_{use_case}", "model_entry_details.json"
        ),
        "w",
    ) as f:
        json.dump(model_entry_details, f, indent=4)

    return model_entry_details
