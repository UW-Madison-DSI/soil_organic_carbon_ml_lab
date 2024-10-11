import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Get MapTiler API key from secrets
maptilerkey = st.secrets['MAP_TILER_KEY']


def histogram_var(data, var):
    """
    Generate and display a histogram for a specified variable.
    Args:
        data (DataFrame): The dataset containing the variable.
        var (str): The variable to plot.
    """
    var_ref = (var.replace('_', ' ')
               .replace('norm', '')
               .replace('mean', '')
               .replace('om', 'Organic matter')
               .replace('bd', 'Bulk density')
               .capitalize())

    fig = px.histogram(data, x=var, title=f"{var_ref} - Distribution")
    st.plotly_chart(fig)


def trend_time_series(dframe, depth_c, select_state):
    """
    Display a time series trend for soil organic carbon with optional state and depth filters.
    Args:
        dframe (DataFrame): The dataset containing soil organic carbon data.
        depth_c (str): Selected soil depth range.
        select_state (str): Selected state for filtering.
    """
    df = dframe

    if 'ALL' not in depth_c:
        df = df[df['depth_cm'] == int(depth_c.split('-')[-1])]

    if select_state != 'ALL':
        df = df[df['State_Name'] == select_state]

    # Group data by year and state, and compute mean and SEM for soil organic carbon
    df_aggregated = df.groupby(['State_Name', 'year']).agg(
        mean_soil_organic_carbon=('soil_organic_carbon', 'median'),
        sem_soil_organic_carbon=('soil_organic_carbon', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        count=('soil_organic_carbon', 'count')
    ).reset_index()

    # Calculate confidence intervals
    df_aggregated['t_critical'] = df_aggregated['count'].apply(lambda x: t.ppf(0.975, x - 1))
    df_aggregated['ci_lower'] = df_aggregated['mean_soil_organic_carbon'] - df_aggregated['t_critical'] * df_aggregated[
        'sem_soil_organic_carbon']
    df_aggregated['ci_upper'] = df_aggregated['mean_soil_organic_carbon'] + df_aggregated['t_critical'] * df_aggregated[
        'sem_soil_organic_carbon']

    # General trend analysis using Exponential Smoothing
    df_yearly = df.groupby('year').agg({'soil_organic_carbon': 'median'}).reset_index()
    df_yearly['EWMA'] = df_yearly['soil_organic_carbon'].ewm(alpha=0.5).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_yearly['year'], y=df_yearly['soil_organic_carbon'], mode='lines+markers', name='Observed'))
    fig.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['EWMA'], mode='lines', name='EWMA'))

    fig.update_layout(
        title=f'Soil Organic Carbon Trend - {depth_c.split("-")[-1]} cm Depth',
        xaxis_title='Year',
        yaxis_title='Soil Organic Carbon',
        xaxis=dict(rangeslider=dict(visible=False), autorange=True)
    )

    st.plotly_chart(fig)


def init():
    """
    Initialize and display a heatmap of soil organic carbon values.
    """
    turkey_coord = [39.653098, -99.101648]
    turkey_map_normal = folium.Map(location=turkey_coord, zoom_start=5.5)
    df = pd.read_parquet('data/sample_soc_observations/final_conus_v2.parquet', engine='pyarrow')

    heat_data = df[['latitude', 'longitude', 'soil_organic_carbon']].values.tolist()
    HeatMap(data=heat_data, radius=5).add_to(turkey_map_normal)

    st_folium(turkey_map_normal, width=700, height=500)


def map_customization_function():
    """
    Display a map with a rectangle around a specific point.
    """
    st.title("Map with Rectangles around Points")

    lat, lon = 40.7128, -74.0060  # Example: New York City
    delta = 0.01  # Size of the rectangle
    bounding_box = [[lat - delta, lon - delta], [lat + delta, lon + delta]]

    m = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Rectangle(bounds=bounding_box, color='blue', fill=True, fill_opacity=0.2).add_to(m)
    folium.Marker([lat, lon], popup="New York City").add_to(m)

    st_folium(m, width=700, height=500)
