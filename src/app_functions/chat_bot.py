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

# Function to get context from GPT-3.5 using the new API
def get_context_from_model(question, context):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    response = qa_pipeline(question=question, context=context)
    return response['answer']


# Main function of the application
def chat_contextualized():
    #st.write("Context Retriever with Transformers")

    parquet_file_path = '../../data/sample_soc_observations/final_conus_v2.parquet'

    # Load and preprocess the data
    df = load_data(parquet_file_path)
    texts = preprocess_data(df)

    # Combine all texts to create a single context
    context = " ".join(texts[:1000])  # You might need to adjust this depending on your context size

    # Predefined questions
    questions = [
        "What is the average soil organic carbon content in the dataset?",
        "Which region has the highest soil organic carbon content?",
        "How does the soil organic carbon content vary across different regions?"
    ]

    selected_question = st.selectbox("Select a question", questions)

    if st.button("Get Answer"):
        if selected_question:
            # Get the answer from the model
            answer = get_context_from_model(selected_question, context)
            st.write("Answer:")
            st.write(answer)
        else:
            st.error("Please select a question.")
