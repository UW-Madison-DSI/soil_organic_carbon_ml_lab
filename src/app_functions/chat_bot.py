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
        "How the soil organic carbon content can help my land?",
        "Which region has the highest soil organic carbon content?",
        "How does the soil organic carbon content vary across different regions?"
    ]

    user_input = st.selectbox("Select a question", questions)

    if st.button("Send"):
        # Generate response
        st.write("In progress")
        #response = generate_response(user_input)
        st.text_area("Dialog-GPT:", value='In progress', height=200, max_chars=None)
