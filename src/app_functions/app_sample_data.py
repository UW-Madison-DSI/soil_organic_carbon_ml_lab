import os
import io
import base64
import tempfile
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import altair as alt
from scipy.stats import t
import rasterio
from PIL import Image
from rasterio import warp
from rasterio.plot import show

import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
from shapely.geometry import Point, Polygon
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

import folium
from folium import CircleMarker, TileLayer
import branca.colormap as cm

import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import folium_static

import emoji

from src.app_functions.functions import add_time_series_prediction_with_predictors
from src.app_functions.map_layers import *

@st.cache_data
def load_data_conus():
    """
    Load CONUS data from a Parquet file.
    Returns:
        DataFrame: The loaded dataset.
    """
    return pd.read_parquet('data/sample_soc_observations/final_conus_v2.parquet', engine='pyarrow')


states_list = [
    'ALL', 'ALABAMA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT',
    'DELAWARE', 'FLORIDA', 'GEORGIA', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA',
    'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS',
    'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA',
    'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK',
    'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON',
    'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
    'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON',
    'WEST VIRGINIA', 'WISCONSIN', 'WYOMING']

features = [
        'label', 'depth_cm', 'min_temperature', 'max_temperature', 'total_precipitation',
        'land_cover', 'land_use', 'clay_mean', 'silt_mean', 'sand_mean', 'bd_mean',
        'dem', 'slope', 'aspect', 'hillshade', 'om_mean', 'ph_mean'
        ]

def observed_soil_dynamic_properties() -> None:
    """
    Display the Sampling Locations of Soil Organic Carbon and Soil Dynamic Properties.
    This function generates an interactive map and relevant statistics based on user input.
    """
    st.title("Sampling Locations of Soil Organic Carbon and Soil Dynamic Properties.")

    # Load dataset
    data = load_data_conus()

    # Sidebar options for map style, state, and depth selection
    map_style = st.sidebar.selectbox('Select the map style :world_map:', ('OpenStreetMap', 'USGS Imagery'))
    select_state = st.sidebar.selectbox("State :round_pushpin:", states_list, key=4)
    depthslist = ['0-5', '5-15', '15-30', '30-60', '60-100', '100-200', 'ALL']
    depth_c = st.sidebar.selectbox("Soil Depth (cm)", depthslist, key=2)

    uploaded_file = None
    if uploaded_file is not None:
        try:
            # Adding derived column 'soil_organic_carbon_stocks'
            if 'soil_organic_carbon_predictions' in data.columns:
                data['soil_organic_carbon_stocks'] = (
                        data['soil_organic_carbon_predictions'] * data['depth_cm'] * data['bd_mean']
                )
                features1 = ['soil_organic_carbon_stocks', 'label'] + features[1:]
            else:
                features1 = features[1:] + ['label']
        except Exception as e:
            st.error(e)
    else:
        # Loading data and preparing feature columns
        data_load_state = st.text("Loading data...")
        data = load_data_conus()
        data_load_state.text("")

        # Add 'soil_organic_carbon_stocks' column
        data['soil_organic_carbon_stocks'] = (
                data['soil_organic_carbon'] * data['depth_cm'] * data['bd_mean']
        )
        features1 = [
            'land_cover', 'land_cover_class', 'land_use', 'land_use_class', 'total_precipitation',
            'min_temperature', 'mean_temperature', 'max_temperature', 'dem', 'slope', 'aspect',
            'hillshade', 'bd_mean', 'clay_mean', 'om_mean', 'ph_mean', 'sand_mean', 'silt_mean',
            'soil_organic_carbon', 'soil_organic_carbon_stocks', 'label'
        ]
    features1.sort()

    # Filter data by year
    data['date'] = pd.to_datetime(data['year'], format='%Y')
    data = data.set_index('date')

    earliest_year = data["year"].min()
    latest_year = data["year"].max()
    min_year, max_year = st.slider(
        "Year Range of Data",
        min_value=int(earliest_year),
        max_value=int(latest_year),
        value=[int(earliest_year), int(latest_year)]
    )

    filtered_data = data[(data["year"] >= min_year) & (data["year"] <= max_year)]

    try:
        # Load state shapefile and filter by selected state
        states = gpd.read_file('data/states_shape/States_shapefile.shp')
        gdf = gpd.GeoDataFrame(
            filtered_data,
            geometry=gpd.points_from_xy(filtered_data.longitude, filtered_data.latitude),
            crs="EPSG:4326"
        )
        if select_state != 'ALL':
            state_gdf = states[states['State_Name'] == select_state]
            filtered_data0 = gpd.sjoin(gdf, state_gdf, how="inner")
        else:
            filtered_data0 = filtered_data

        # Sidebar for selecting soil properties to visualize
        soil_properties_c = st.sidebar.selectbox("Soil Properties", features1, key=3)
        soil_properties = soil_properties_c.replace(" (observed)", "")

        # Filter data by depth
        terraincls = ['bd_mean', 'clay_mean', 'om_mean', 'ph_mean', 'sand_mean', 'silt_mean']
        depth_filter = (
                    filtered_data0['depth_cm'] == int(depth_c.split('-')[-1])) if soil_properties in terraincls else (
                    filtered_data0['depth_cm'] == 5)
        filtered_data1 = filtered_data0[depth_filter]

        # Display total samples
        st.write("Total sample data: ", len(filtered_data1))

        # Map visualization
        map_observations(filtered_data1, soil_properties, map_style)

        if 'State_Name' in filtered_data1.columns:
            # Split the screen into two columns for time series and histogram
            col1, col2 = st.columns([3, 1])
            with col1:
                trend_time_series(filtered_data0, depth_c, 'ALL')
            with col2:
                histogram_var(filtered_data1, soil_properties)
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                scatterplot_var(filtered_data1, soil_properties, 'soil_organic_carbon')
            with col2:
                histogram_var(filtered_data1, soil_properties)


        # In your main app function (e.g., `main()` or where your Streamlit UI is defined)
        st.sidebar.header("Time Series Analysis")
        if st.button("Run Time Series Prediction"):
            # Assuming 'filtered_data' is a DataFrame filtered by map bounds
            add_time_series_prediction_with_predictors(
                    filtered_data1,
                    [soil_properties]  # Use a variable from your filters
                )

    except Exception as e:
        st.write(f"Error: {e}")
