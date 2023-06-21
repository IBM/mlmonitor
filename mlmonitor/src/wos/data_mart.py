# SPDX-License-Identifier: Apache-2.0
from ibm_watson_openscale.supporting_classes.enums import DatabaseType
from ibm_watson_openscale.supporting_classes import (
    DatabaseConfigurationRequest,
    LocationSchemaName,
    PrimaryStorageCredentialsLong,
)
from ibm_watson_openscale import APIClient


def get_datamart_ids(
    wos_client: APIClient,
):
    return [dm.metadata.id for dm in wos_client.data_marts.list().result.data_marts]


def create_datamart(
    wos_client: APIClient, schema_name: str = None, db_credentials: dict = None
):
    data_marts = get_datamart_ids(wos_client=wos_client)
    if len(data_marts) == 0:
        if db_credentials is not None:
            if schema_name is None:
                raise ValueError("Please specify the SCHEMA_NAME and rerun the cell")

            print("Setting up external datamart")
            added_data_mart_result = wos_client.data_marts.add(
                background_mode=False,
                name="WOS Data Mart",
                description="Data Mart created by WOS tutorial notebook",
                database_configuration=DatabaseConfigurationRequest(
                    database_type=DatabaseType.POSTGRESQL,
                    credentials=PrimaryStorageCredentialsLong(
                        hostname=db_credentials["hostname"],
                        username=db_credentials["username"],
                        password=db_credentials["password"],
                        db=db_credentials["database"],
                        port=db_credentials["port"],
                        ssl=True,
                        sslmode=db_credentials["sslmode"],
                        certificate_base64=db_credentials["certificate_base64"],
                    ),
                    location=LocationSchemaName(schema_name=schema_name),
                ),
            ).result

        else:
            print("Setting up internal Data Mart")
            added_data_mart_result = wos_client.data_marts.add(
                background_mode=False,
                name="WOS Data Mart",
                description="Data Mart created in PyCharm client",
                internal_database=True,
            ).result

        return added_data_mart_result.metadata.id

    else:
        print(
            f"found {len(data_marts)} data mart :  {data_marts} , Using existing Data Mart id {data_marts[0]}"
        )
        return data_marts[0]
