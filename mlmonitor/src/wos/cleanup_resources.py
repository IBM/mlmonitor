# SPDX-License-Identifier: Apache-2.0
from boto3.session import Session as boto_Session
from typing import Optional

from mlmonitor.src import API_KEY, ENV, USERNAME, AUTH_ENDPOINT, logger, VERIFY_CP4D_SSL
from mlmonitor.src.wos import wos_client
from mlmonitor.src.wos.subscription import delete_subscription
from mlmonitor.src.factsheets.utils import (
    FactsheetHelpers,
    get_model_id_by_deployment_name,
    get_model_id_by_model_name,
)
from mlmonitor.src.wml import WML_URL
from mlmonitor.src.wml.utils import get_model_uid_by_name


def delete_deployment(
    deployment_name: str,
    model_entry_id: str,
    catalog_id: str,
    apikey: str = API_KEY,
    deployment_target: str = "aws",
) -> dict:
    """
    Deletes all resources for deployment <deployment_name> in : Model serving (Sagemaker,WML,Azure...) , WOS , Factsheets (model asset)

    :param deployment_name:str: EndpointName of AWS online inference endpoint or WML deployment name.
    :param model_entry_id:str: BM AI Factsheets Model use case identifier where the model deployment was created
    :param catalog_id:str: IBM AI Factsheets catalog identifier  where the model deployment was created
    :param apikey:str=API_KEY: IBM Cloud or CP4D on premise API key to use for authentication
    :param deployment_target:str=aws: target model serving environment where model should be deleted
    :return: Dictionary of status for each resource type that should be deleted : Model serving (Sagemaker,WML,Azure...)  , WOS , Factsheets model asset
    count of Number of resources deleted
    """

    assert deployment_target in {
        "wml",
        "aws",
        "azure",
    }, "Deployment target should be wml ,aws or azure"
    assets_deleted, wos_deleted, ep_deleted = (0, 0, 0)

    ####################################
    # CLEANUP MODEL SERVING RESOURCES  #
    ####################################
    logger.info(f"deleting [{deployment_target}] resources for [{deployment_name}]...")

    if deployment_target == "aws":
        #######################################################################################
        # CLEANUP AWS RESOURCES (Endpoint, EndpointConfig , Model) + External model asset(s)  #
        #######################################################################################
        from mlmonitor.src import aws_credentials
        from mlmonitor.src.aws.scoring import delete_deployment_resources

        session = boto_Session(**aws_credentials)
        sagemaker_client = session.client("sagemaker")
        ep_deleted = delete_deployment_resources(
            sagemaker_client=sagemaker_client, endpoint_name=deployment_name
        )

    elif deployment_target == "azure":
        #############################################
        # CLEANUP Azure RESOURCES (Endpoint, Model) #
        #############################################
        from mlmonitor.src.azure import AZ_WORKSPACE
        from mlmonitor.src.azure.deployment import delete_az_resources

        ep_deleted = delete_az_resources(
            deployment_name=deployment_name, workspace=AZ_WORKSPACE, logger=logger
        )

    ####################################
    # CLEANUP External model asset(s)  #
    ####################################
    if deployment_target in {"aws", "azure"}:

        assets_deleted = cleanup_external_model_assets(
            deployment_name=deployment_name,
            api_key=apikey,
            wkc_catalog_id=catalog_id,
            model_use_case_id=model_entry_id,
            cpd_username=USERNAME,
            cpd_url=AUTH_ENDPOINT,
        )

    elif deployment_target == "wml":
        #############################################################
        # CLEANUP WML RESOURCES (Deployment, Model Asset)           #
        #############################################################
        from mlmonitor.src.wml import wml_client, WML_SPACE_ID

        wml_client.set.default_space(WML_SPACE_ID)
        ep_deleted = 0
        if model_asset_id := get_model_uid_by_name(
            wml_client=wml_client, model_name=deployment_name
        ):
            from mlmonitor.src.wml.scoring import delete_deployment_resources
            from ibm_aigov_facts_client import AIGovFactsClient, CloudPakforDataConfig

            aigov_params = {
                "experiment_name": "cleanup",
                "enable_autolog": False,
                "set_as_current_experiment": True,
                "external_model": False,
                "container_type": "space",
                "container_id": WML_SPACE_ID,
            }

            if ENV == "saas":
                aigov_params["api_key"] = API_KEY
            elif ENV == "prem":
                aigov_params["cloud_pak_for_data_configs"] = CloudPakforDataConfig(
                    service_url=WML_URL, username=USERNAME, api_key=API_KEY
                )
            else:
                raise ValueError(
                    f"ENV set to '{ENV}'.Value should be set to 'saas' (IBM Cloud) or 'prem' (On premise cluster)"
                )

            facts_client = AIGovFactsClient(**aigov_params)

            # remove assets in space only for WML models ("external_model": False)
            wml_model = facts_client.assets.get_model(model_id=model_asset_id)
            wml_model.remove_tracking_model_usecase()

            facts_client.assets.remove_asset(asset_id=model_asset_id)
            ep_deleted += 1
            assets_deleted = ep_deleted + 1
            # Not need to explicitly delete deployment and model asset through wml client since assets.remove_asset does it.
            # ep_deleted = delete_deployment_resources(wml_client=wml_client, endpoint_name=deployment_name)
            # wml_client.model_definitions.delete(model_definition_uid=model_asset_id)

    else:
        raise NotImplementedError("deployment_target should be aws ,wml or azure")

    ########################################
    # CLEANUP WOS RESOURCES (subscription) #
    ########################################
    if wos_client:  # wos must be instantiated
        wos_deleted += delete_subscription(
            wos_client=wos_client, deployment_name=deployment_name
        )

    return {
        "wos": wos_deleted,
        "factsheets": assets_deleted,
        "serving": ep_deleted,
    }


def cleanup_external_model_assets(
    deployment_name: str,
    api_key: str,
    wkc_catalog_id: str,
    model_use_case_id: str,
    delete_dev: bool = False,
    cpd_url: Optional[str] = None,
    cpd_username: Optional[str] = None,
) -> int:
    """
    removes the model asset from the AI factsheets model use case and delete model asset with one deployment.

        1) Retrieving all models in a given WKC catalog (by default, it will use the one specified in your config file).
        2) Finding and removing any model assets associated with a given deployment name.
        used to remove all deployments for a given external model in AI FactSheets .

    :param deployment_name:str: Identify the model asset to delete based on its deployment name
    :param api_key:str: CPD4 API Key
    :param wkc_catalog_id:str: AI FactSheets catalog ID where the external model asset is stored
    :param model_use_case_id:str: Identify the AI FactSheets model use case ID in the catalog
    :param delete_dev:bool=False flag to indicate if asset in Develop state should be deleted
    :param cpd_url:Optional[str]=None: CP4D url (only required for on prem environment ENV='prem')
    :param cpd_username:Optional[str]=None: CP4D username (only required for on prem environment ENV='prem')
    :return: number of successfully assets deleted
    """
    delete_dev = True
    deleted_assets = 0
    assets = []

    fs_helpers = FactsheetHelpers(
        api_key=api_key,
        container_type="catalog",
        container_id=wkc_catalog_id,
        model_entry_id=model_use_case_id,
        cpd_url=cpd_url,
        username=cpd_username,
        env=ENV,
    )

    models = fs_helpers.get_models(verify=VERIFY_CP4D_SSL)

    if delete_dev:
        assets.append(
            get_model_id_by_model_name(
                models=models, model_name=deployment_name, state="development", key=None
            )
        )

    # all deployments are assigned same name as trained model asset
    assets.append(
        get_model_id_by_deployment_name(
            models=models, deployment_name=deployment_name, key=None
        )
    )

    for asset in assets:
        if asset:
            model_asset_id = asset.get("id")
            model_container_id = asset.get("container_id")
            container_type = asset.get("container_type")

            # unlink model from model use case (remove tracking) equivalent to remove_tracking_model_usecase()
            res = fs_helpers.unlink_model_asset_from_entry(
                model_asset_id=model_asset_id,
                container_type=container_type,
                container_id=model_container_id,
                verify=VERIFY_CP4D_SSL,
            )

            if res == 204:
                logger.info(
                    f"Model asset [{model_asset_id}] for deployment [{deployment_name}] unlinked from catalog_id [{wkc_catalog_id}] model_use_case_id [{model_use_case_id}]"
                )
                # delete Asset
                res = fs_helpers.delete_asset(
                    model_asset_id=model_asset_id,
                    container_type=container_type,
                    container_id=model_container_id,
                    verify=VERIFY_CP4D_SSL,
                )
                if res == 204:
                    deleted_assets += 1
                    logger.info(
                        f"Model asset [{model_asset_id}] for deployment [{deployment_name}] Deleted"
                    )
                else:
                    logger.info(
                        f"Failed to delete Model asset [{model_asset_id}] for deployment [{deployment_name}] => {res}"
                    )
            else:
                logger.info(
                    f"Failed to unlinked  asset [{model_asset_id}] for deployment [{deployment_name}] from catalog_id [{wkc_catalog_id}] model_use_case_id [{model_use_case_id}] => {res}"
                )
        else:
            logger.info(
                f"No model asset found for deployment [{deployment_name}] in catalog_id [{wkc_catalog_id}] model_use_case_id [{model_use_case_id}]"
            )

    return deleted_assets


# TODO
#  with AIGOV Client new API calls asset can be deleted with remove_asset replace FactsheetHelpers :
#
#  model_entry = facts_client.assets.list_model_usecases(catalog_id=catalog_id)
#  physical_models = model_entry[0].get('entity').get('modelfacts_global').get('physical_models')
#  asset_ids = [asset.get('id') for asset in physical_models]
#
# for model_asset_id in asset_ids:
#     external_model = facts_client.assets.get_model(model_id=model_asset_id)
#     external_model.remove_tracking_model_usecase()
#     facts_client.assets.remove_asset(asset_id=model_asset_id)
