import ee
import streamlit as st
from google.oauth2 import service_account

SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
]

credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets),
    scopes=SCOPES
)

ee.Initialize(
    credentials,
    project=st.secrets["project_id"]
)
