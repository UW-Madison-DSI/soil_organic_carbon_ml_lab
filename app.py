import os
import io
import base64
from tempfile import NamedTemporaryFile
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
import altair as alt
from scipy.stats import t
import rasterio
from rasterio import warp
from rasterio.plot import show
from PIL import Image

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
#from transformers import AutoModelForCausalLM, AutoTokenizer
#import torch

from functions import *

@st.cache_data
def load_data_conus():
    dta = pd.read_parquet('data/final_conus_v2.parquet', engine='pyarrow')
    return dta

def local_css(file_name):
    with open(file_name) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

states_list = ['ALL','ALABAMA',
 'ARIZONA',
 'ARKANSAS',
 'CALIFORNIA',
 'COLORADO',
 'CONNECTICUT',
 'DELAWARE',
 #'DISTRICT OF COLUMBIA',
 'FLORIDA',
 'GEORGIA',
 'IDAHO',
 'ILLINOIS',
 'INDIANA',
 'IOWA',
 'KANSAS',
 'KENTUCKY',
 'LOUISIANA',
 'MAINE',
 'MARYLAND',
 'MASSACHUSETTS',
 'MICHIGAN',
 'MINNESOTA',
 'MISSISSIPPI',
 'MISSOURI',
 'MONTANA',
 'NEBRASKA',
 'NEVADA',
 'NEW HAMPSHIRE',
 'NEW JERSEY',
 'NEW MEXICO',
 'NEW YORK',
 'NORTH CAROLINA',
 'NORTH DAKOTA',
 'OHIO',
 'OKLAHOMA',
 'OREGON',
 'PENNSYLVANIA',
 'RHODE ISLAND',
 'SOUTH CAROLINA',
 'SOUTH DAKOTA',
 'TENNESSEE',
 'TEXAS',
 'UTAH',
 'VERMONT',
 'VIRGINIA',
 'WASHINGTON',
 'WEST VIRGINIA',
 'WISCONSIN',
 'WYOMING']

features = ['label','depth_cm', 'min_temperature', 'max_temperature',
            'total_precipitation','land_cover', 'land_use', 'clay_mean',
            'silt_mean', 'sand_mean','bd_mean', 'dem', 'slope', 'aspect',
            'hillshade', 'om_mean', 'ph_mean']


def soil_dynamic_properties():
    st.title("Soil Organic Carbon Tool.")
    data = load_data_conus()

    map_style = st.sidebar.selectbox(
        'Select the map style :world_map:',
        ('OpenStreetMap',  # 'CartoDB Positron', 'Stamen Terrain',
         'USGS Imagery'  # , 'Canada Weather Radar'
         )
    )
    select_state = st.sidebar.selectbox("State :round_pushpin:", states_list, key=4)

    depthslist=['0-5', '5-15', '15-30', '30-60', '60-100', '100-200','ALL']
    depth_c = st.sidebar.selectbox("Soil Depth (cm)", depthslist, key=2)

    uploaded_file = None
    label = False
    if uploaded_file is not None:
        label = True
        try:
            data = None

            if 'soil_organic_carbon_predictions' in data.columns:
                features1 = ['soil_organic_carbon_predictions', 'label'] + features[1:]
                data['soil_organic_carbon_stocks'] = data['soil_organic_carbon_predictions'] * data['depth_cm'] * data[
                    'bd_mean']
                features1 = ['soil_organic_carbon_stocks', 'label'] + features[1:]
            else:
                features1 = features[1:] + ['label']

        except Exception as e:
            st.error(e)
    else:
        data_load_state = st.text("Loading data...")
        data = load_data_conus()
        data_load_state.text("")
        data['soil_organic_carbon_stocks'] = data['soil_organic_carbon'] * data['depth_cm'] * data['bd_mean']
        features1 = ['land_cover', 'land_cover_class', 'land_use', 'land_use_class',
                     'total_precipitation', 'min_temperature', 'mean_temperature', 'max_temperature',
                     'dem', 'slope', 'aspect', 'hillshade', 'bd_mean', 'clay_mean', 'om_mean', 'ph_mean',
                     'sand_mean', 'silt_mean', 'soil_organic_carbon', 'soil_organic_carbon_stocks', 'label']

    data['date'] = pd.to_datetime(data['year'], format='%Y')

    data = data.set_index('date')


    earliest_year = data["year"].min()
    latest_year = data["year"].max()
    min_year, max_year = st.slider(
        "Year Range of Data",
        min_value=int(earliest_year),
        max_value=2020,
        value=[int(earliest_year), int(latest_year)],
    )

    filtered_data = data[data["year"] >= min_year - 1]
    filtered_data = filtered_data[filtered_data["year"] <= max_year]
    try:

        states = gpd.read_file('data/states_shape/States_shapefile.shp')
        if select_state is not None and select_state != 'ALL':
            gdf = gpd.GeoDataFrame(filtered_data,
                                   geometry=gpd.points_from_xy(filtered_data.longitude, filtered_data.latitude))
            state_gdf = states[states['State_Name'] == select_state]
            # Perform spatial join
            filtered_data0 = gpd.sjoin(gdf, state_gdf, how="inner")
        else:
            filtered_data0 = filtered_data

        # st.write(filtered_data0.columns)
        soil_properties_c = st.sidebar.selectbox("Soil Properties",
                                             list(set(features1))
                                              # 'soil_organic_carbon_predictions','soil_organic_carbon_stocks'
                                              , key=3)
        soil_properties = soil_properties_c.replace(" (observed)", "")
        terraincls = ['bd_mean', 'clay_mean', 'om_mean', 'ph_mean', 'sand_mean', 'silt_mean']
        # depth_c = st.radio("Depth (cm)", ('0-5', '5-15', '15-30', '30-60', '60-100', '100-200', 'ALL'))
        if soil_properties in terraincls:
            depth_c1 = depth_c
            # st.sidebar.radio("Depth (cm)", ('0-5', '5-15', '15-30', '30-60', '60-100', '100-200', 'ALL'), key=1)
            filtered_data1 = filtered_data0
            if 'ALL' not in depth_c1:
                depth = depth_c1.split('-')[-1]
                filtered_data1 = filtered_data0[filtered_data0['depth_cm'] == int(depth)]
        else:
            filtered_data1 = filtered_data0[filtered_data0['depth_cm'] == 5]

        # st.write("Soil Profiles: ", len(filtered_data1['soil_id'].drop_duplicates()))
        st.write("Total sample data: ", len(filtered_data1))


        map1(filtered_data1, soil_properties, map_style)
        col1, col2 = st.columns([2, 1])
        with col1:
            trend_time_series(filtered_data0, depth_c, 'ALL')
        with col2:
            histogram_var(filtered_data1, soil_properties, label)

    except Exception as e:
        st.write(f"{e}")


def soc_prediction():
    # Streamlit app
    #st.title('SOC prediction')

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload my .shp file", type="shp")
    tmp_path = 'data/prcp_1850 copy.tif'
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Create and display the map
    if tmp_path:
        img, bounds = create_map_with_geotiff(tmp_path)
        map_layers(img, bounds)
    else:
        map_layers(False, False)
    #map=map22(tmp_path)
    #folium_static(map)
    #upload1()

def login():
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == st.secrets['PASSW']:
            st.session_state.authenticated = True
        else:
            st.error("Incorrect password")

@st.cache_data()
def load_model():
    model_name = "microsoft/DialoGPT-small"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer





def generate_response(prompt):
    model, tokenizer = load_model()
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def dialog():
    st.write("### Chat with DialoGPT")

    # User input
    questions = [
        "Which region has the highest soil organic carbon content?",
        "How does the soil organic carbon content vary across different regions?"
    ]

    user_input = st.selectbox("Select a question", questions)


    if st.button("Send"):
        # Generate response
        st.write("In progress")
        #response = generate_response(user_input)
        #st.text_area("Dialog-GPT:", value=response, height=200, max_chars=None)


def main1():
    local_css("frontend/css/streamlit.css")
    st.sidebar.title("Functionalities")

    sidebar_object = st.sidebar.radio('Please choose the option', ("SOC Assistant",'My Data'), key=100)

    if sidebar_object == 'SOC Assistant':
        soc_prediction()
        #dialog()

    else:
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False

        if st.session_state.authenticated==False:
            login()
        else:
            soil_dynamic_properties()


if __name__ == "__main__":
    main1()