import ee
import streamlit as st
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets)
)

ee.Initialize(
    credentials,
    project=st.secrets["project_id"]
)
