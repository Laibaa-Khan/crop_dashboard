import ee
import streamlit as st
from google.oauth2 import service_account

# Load service account credentials from Streamlit Secrets
service_account_info = {
    "type": st.secrets["type"],
    "project_id": st.secrets["project_id"],
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"],
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": st.secrets["auth_uri"],
    "token_uri": st.secrets["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["client_x509_cert_url"],
    "universe_domain": "googleapis.com"
}

# Create credentials object
credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)

# Initialize Earth Engine
ee.Initialize(
    credentials,
    project=st.secrets["project_id"]
)
