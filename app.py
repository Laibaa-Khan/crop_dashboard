import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee

from gee_backend import *

ee.Initialize(project="bubbly-sentinel-486808-v7")

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Crop AI Dashboard",
    layout="wide"
)

# ==========================
# CSS
# ==========================

st.markdown("""
<style>

.stApp {
    background-color: #F4FFF4;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1B5E20,#2E7D32);
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white;
}

.stSelectbox div[data-baseweb="select"] {
    color: black !important;
}

div.stButton > button {
    background-color: #2E7D32;
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background-color: #1B5E20;
}

div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ==========================
# TITLE
# ==========================

st.markdown("""
<h1 style='text-align:center;color:#1B5E20'>
🌾 Pakistan Crop Classification Platform
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<center>
AI-based Crop Classification using Sentinel-2,
Random Forest and SVM
</center>
""", unsafe_allow_html=True)

st.divider()

# ==========================
# DEFAULT DATASET
# ==========================

default_table = ee.FeatureCollection(
    "projects/bubbly-sentinel-486808-v7/assets/Pakistan_Agricultural"
)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("⚙️ Control Panel")

dataset_mode = st.sidebar.radio(
    "Dataset Source",
    [
        "Default Dataset",
        "Upload Dataset"
    ]
)

province = st.sidebar.selectbox(
    "Province",
    [
        "Punjab",
        "Sindh",
        "Khyber Pakhtunkhwa",
        "Balochistan"
    ]
)

crop = st.sidebar.selectbox(
    "Crop",
    [
        "Wheat",
        "Rice",
        "Cotton",
        "Maize",
        "Sugarcane"
    ]
)

model_choice = st.sidebar.radio(
    "Model",
    [
        "Random Forest",
        "SVM",
        "Both"
    ]
)

uploaded_file = None

if dataset_mode == "Upload Dataset":

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

if "run_model" not in st.session_state:
    st.session_state.run_model = False

if st.sidebar.button("🚀 Run Classification"):
    st.session_state.run_model = True

# ==========================
# PROJECT INFO
# ==========================

with st.expander("Project Information"):

    st.write("""
    • Google Earth Engine

    • Sentinel-2 Imagery

    • Random Forest

    • Support Vector Machine

    • Province-wise Crop Mapping

    • User Uploaded Dataset

    """)

# ==========================
# RUN
# ==========================

if st.session_state.run_model:

    try:

        with st.spinner("Processing Satellite Data..."):

            region = get_region(province)

            start, end = get_season(crop)

            # ==========================
            # DATASET
            # ==========================

            if dataset_mode == "Default Dataset":

                table = default_table

            else:

                if uploaded_file is None:

                    st.error(
                        "Please upload a CSV file."
                    )

                    st.stop()

                df = pd.read_csv(uploaded_file)

                table = csv_to_ee(df)

            train = get_training(
                table,
                region,
                crop,
                province
            )

            total_samples = train.size().getInfo()

            # ==========================
            # MAP
            # ==========================

            m = folium.Map(
                location=[30.5,69.3],
                zoom_start=6
            )

            # ==========================
            # RF
            # ==========================

            if model_choice in [
                "Random Forest",
                "Both"
            ]:

                rf_map, rf_acc = run_rf(
                    region,
                    train,
                    start,
                    end
                )

                rf_tile = rf_map.getMapId({
                    "min":0,
                    "max":1,
                    "palette":[
                        "white",
                        "green"
                    ]
                })

                folium.TileLayer(
                    tiles=rf_tile[
                        "tile_fetcher"
                    ].url_format,
                    name="Random Forest",
                    overlay=True,
                    attr="GEE"
                ).add_to(m)

            # ==========================
            # SVM
            # ==========================

            if model_choice in [
                "SVM",
                "Both"
            ]:

                svm_map, svm_acc = run_svm(
                    region,
                    train,
                    start,
                    end
                )

                svm_tile = svm_map.getMapId({
                    "min":0,
                    "max":1,
                    "palette":[
                        "white",
                        "blue"
                    ]
                })

                folium.TileLayer(
                    tiles=svm_tile[
                        "tile_fetcher"
                    ].url_format,
                    name="SVM",
                    overlay=True,
                    attr="GEE"
                ).add_to(m)

            folium.LayerControl().add_to(m)

        # ==========================
        # METRICS
        # ==========================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Province",
                province
            )

        with col2:
            st.metric(
                "Crop",
                crop
            )

        with col3:
            st.metric(
                "Samples",
                total_samples
            )

        if model_choice == "Both":

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "RF Accuracy",
                    f"{rf_acc*100:.2f}%"
                )

            with c2:
                st.metric(
                    "SVM Accuracy",
                    f"{svm_acc*100:.2f}%"
                )

        elif model_choice == "Random Forest":

            st.metric(
                "RF Accuracy",
                f"{rf_acc*100:.2f}%"
            )

        elif model_choice == "SVM":

            st.metric(
                "SVM Accuracy",
                f"{svm_acc*100:.2f}%"
            )

        st.subheader(
            "🗺️ Classification Map"
        )

        st_folium(
            m,
            width=1200,
            height=650
        )

        report = f"""
Province: {province}

Crop: {crop}

Samples: {total_samples}
"""

        if model_choice in [
            "Random Forest",
            "Both"
        ]:
            report += (
                f"\nRF Accuracy: "
                f"{rf_acc*100:.2f}%"
            )

        if model_choice in [
            "SVM",
            "Both"
        ]:
            report += (
                f"\nSVM Accuracy: "
                f"{svm_acc*100:.2f}%"
            )

        st.download_button(
            "📥 Download Report",
            report,
            "report.txt"
        )

    except Exception as e:

        st.error(str(e))

else:

    st.info(
        "Select options and click Run."
    )

st.divider()

st.markdown("""
<center>

Developed using Streamlit + Google Earth Engine

NED University Project

</center>
""", unsafe_allow_html=True)

