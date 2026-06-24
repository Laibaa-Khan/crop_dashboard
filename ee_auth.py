import ee
import json
import os

from google.oauth2 import service_account

service_account_info = json.loads(
    os.environ["EE_KEY"]
)

credentials = (
    service_account.Credentials
    .from_service_account_info(
        service_account_info
    )
)

ee.Initialize(
    credentials,
    project="bubbly-sentinel-486808-v7"
)