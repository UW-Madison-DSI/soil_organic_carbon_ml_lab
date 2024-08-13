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
import requests
#from transformers import AutoModelForCausalLM, AutoTokenizer
#import torch


maptilerkey = st.secrets['MAP_TILER_KEY']

import seaborn as sns

def trend_time_series(dframe, depth_c, select_state):
    """

    """
    if 'ALL' in depth_c:
        df = dframe
    else:
        df = dframe[dframe['depth_cm']==int(depth_c.split('-')[-1])]

    try:
        if select_state!='ALL':
            df_aggregated = df.groupby(['State_Name', 'year']).agg(
                mean_soil_organic_carbon=('soil_organic_carbon', 'median'),
                sem_soil_organic_carbon=('soil_organic_carbon', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
                count=('soil_organic_carbon', 'count')
            ).reset_index()

            df_pivot = df_aggregated.pivot(index='year', columns='State_Name', values='mean_soil_organic_carbon')
            st.write(df_pivot.head(5))
            st.write(df_aggregated.head(50))
            # Fit SARIMA model for each state
            models = {select_state: sm.tsa.SARIMAX(df_pivot[select_state], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()}
            # Compute confidence intervals for each state-year group
            df_aggregated['t_critical'] = df_aggregated.apply(lambda row: t.ppf(0.975, row['count'] - 1), axis=1)
            df_aggregated['ci_lower'] = df_aggregated['mean_soil_organic_carbon'] - df_aggregated['t_critical'] * df_aggregated[
                'sem_soil_organic_carbon']
            df_aggregated['ci_upper'] = df_aggregated['mean_soil_organic_carbon'] + df_aggregated['t_critical'] * df_aggregated[
                'sem_soil_organic_carbon']

            predictions = {}
            conf_ints = {}
            for state, model in models.items():
                pred = model.get_forecast(steps=2)  # Forecast for 10 years
                st.write("Predictions ", pred.predicted_mean)
                predictions[state] = pred.predicted_mean
                conf_ints[state] = pred.conf_int()
                st.write("here ", pred.conf_int())

            # Plot the results with confidence intervals
            plt.figure(figsize=(14, 7))

            for state in df_pivot.columns:
                plt.plot(df_pivot.index, df_pivot[state], label=f'{state} - Actual')
                plt.plot(predictions[state].index, predictions[state], label=f'{state} - Forecast')
                plt.fill_between(conf_ints[state].index,
                                 conf_ints[state].iloc[:, 0],
                                 conf_ints[state].iloc[:, 1], alpha=0.2, label=f'{state} - CI')

            # Add the confidence intervals of the mean temperature for the historical data
            for state in df_aggregated['State_Name'].unique():
                state_data = df_aggregated[df_aggregated['State_Name'] == state]
                plt.fill_between(state_data['year'],
                                 state_data['ci_lower'],
                                 state_data['ci_upper'], alpha=0.1, label=f'{state} - Historical CI')
            plt.legend()
            plt.xlabel('Year')
            plt.ylabel('Soil Organic Carbon')
            plt.title('Median Soil Organic Carbon across the years')
            st.pyplot(plt)

        df = df.groupby(by='year').agg({'soil_organic_carbon': 'median'})
        df = df.reset_index()

        df['DESAdd'] = ExponentialSmoothing(df['soil_organic_carbon'], trend='add').fit().fittedvalues.shift(-1)
        df['EWMA3'] = df['soil_organic_carbon'].ewm(alpha=2/3, adjust=False).mean()
        df['SES3'] = SimpleExpSmoothing(df['soil_organic_carbon']).fit(smoothing_level=.1,
                                                         optimized=False).fittedvalues.shift(-1)
        fig = go.Figure()

        # Add Sales line
        fig.add_trace(go.Scatter(x=df['year'], y=df['soil_organic_carbon'], mode='lines+markers', name='Obs'))
        # Add Trend line
        fig.add_trace(go.Scatter(x=df['year'], y=df['EWMA3'], mode='lines', name='EWMA3'))
        fig.add_trace(go.Scatter(x=df['year'], y=df['SES3'], mode='lines', name='SES3'))
        # Update layout for autoscaling x-axis
        fig.update_layout(
            title=f'Median Soil organic carbon trend {depth_c.split("-")[-1]} cm depth',
            xaxis_title='year',
            yaxis_title='Soil Organic Carbon',
            xaxis=dict(
                rangeslider=dict(visible=False),
                autorange=True
            )
        )
        # Display the plotly plot in Streamlit
        st.plotly_chart(fig)
    except Exception as e:
        st.error(e)


def histogram_var(data, var):
    '''

    :param data:
    :param var:
    :return:
    '''

    fig = px.histogram(data, x=var, title=f"{var.replace('_',' ').replace('norm','').replace('mean','').replace('om','Organic matter').replace('bd','Bulk density').capitalize()} - Distribution")
    st.plotly_chart(fig)


def upload1():
    '''

    :return:
    '''
    uploaded_file = st.file_uploader("Upload TIFF file", type=["tif", "tiff"], key=1)

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded file
        with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as src:
            st.write(f"Number of bands: {src.count}")
            st.write(f"Width: {src.width}")
            st.write(f"Height: {src.height}")

            st.subheader("Visualizing the TIFF file")
            fig, ax = plt.subplots()
            show(src, ax=ax)
            st.pyplot(fig)


            ###############################################################
            array = src.read()
            bounds = src.bounds

            x1, y1, x2, y2 = src.bounds
            bbox = [(bounds.bottom, bounds.left), (bounds.top, bounds.right)]

            st.title("Plotting maps!")
            # center on Liberty Bell
            m = folium.Map(location=[43.071079, -89.418544], zoom_start=6)

            # add marker for Liberty Bell
            tooltip = "Manilla city"
            folium.Marker(
                [43.071079, -89.418544], popup="This is it!", tooltip=tooltip
            ).add_to(m)

            img = folium.raster_layers.ImageOverlay(
                name="Sentinel 2",
                image=np.moveaxis(array, 0, -1),
                bounds=bbox,
                opacity=0.9,
                interactive=True,
                cross_origin=False,
                zindex=1,
            )
            # folium.Popup("I am an image").add_to(img)
            img.add_to(m)
            folium.LayerControl().add_to(m)

            # call to render Folium map in Streamlit
            folium_static(m)



# Function to load data from the Parquet file
def load_data(parquet_file_path):
    df = pd.read_parquet(parquet_file_path)
    return df


# Function to preprocess the data
def preprocess_data(df):
    # Select relevant numeric and string/class columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    #st.write(numeric_columns)
    string_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Combine the selected columns into a single text column
    relevant_columns = numeric_columns[3:] + string_columns
    df['text'] = df[relevant_columns].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    return df['text'].tolist()


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
