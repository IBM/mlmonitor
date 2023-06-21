# SPDX-License-Identifier: Apache-2.0
import os
import pandas as pd

from mlmonitor.src import USERNAME, AUTH_ENDPOINT
from mlmonitor.src.utils.utils import parse_args
from mlmonitor.src import PROJECT_ROOT, API_KEY, MODEL_ENTRY_ID, CATALOG_ID
from mlmonitor.src.factsheets.utils import FactsheetHelpers


def populate_model_facts(
    model_id: str,
    apikey: str = API_KEY,
    model_entry_id: str = MODEL_ENTRY_ID,
    catalog_id: str = CATALOG_ID,
):
    """
    Creates a model asset schema and populate default values using WKC API
    Parameters
    ----------
    model_id : str
       model asset ID
    apikey : str
       IBM Cloud API key to use for authentication
    model_entry_id : str
       IBM AI Factsheets Model use case identifier to be used to register the deployed model
    catalog_id : str
       IBM AI Factsheets catalog identifier to be used to register the deployed model
    Returns
    -------
    """
    modelfacts_user = os.path.join(
        f"{PROJECT_ROOT}/factsheets_{user_case}/",
        f"custom_model_asset_facts_{user_case}.csv",
    )

    df_modelfacts_values = pd.read_csv(
        f"{PROJECT_ROOT}/factsheets_{user_case}/custom_model_asset_facts_values_{user_case}.csv",
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

    print(f"Applying model facts : {modelfacts_user} ")
    fs_helper.define_custom_model_facts(modelfacts_user)

    fs_helper.set_custom_fact(
        fact_ids=df_modelfacts_values["name"].to_list(),
        fact_values=df_modelfacts_values["default_value"].to_list(),
        model_id=model_id,
        op="add",
    )


if __name__ == "__main__":

    args = parse_args()
    user_case = "churn"
    ibm_apikey = API_KEY
    populate_model_facts(
        model_entry_id=args.model_entry_id,
        catalog_id=args.catalog_id,
        model_id=args.model_id,
        apikey=ibm_apikey,
    )
