import os
import io
import base64
import tempfile
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import plotly.express as px
import altair as alt

import rasterio
from rasterio import warp
from rasterio.plot import show
from PIL import Image

import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
from shapely.geometry import Point, Polygon

import folium
from folium import CircleMarker, TileLayer
import branca.colormap as cm

import requests
import streamlit as st
from streamlit_folium import st_folium

# Project-specific imports
from src.app_functions.functions import *


def mstyle_n_layers(map_style: str) -> tuple[str, list]:
    '''
    Get the Mapbox style and any additional layers based on map style.

    Args:
        map_style (str): The style of the map to be displayed.

    Returns:
        tuple: A tuple containing the mapbox style and any additional layers.
    '''
    layers = []
    mapbox_style = "open-street-map"

    if map_style == 'CartoDB Positron':
        mapbox_style = "carto-positron"
    elif map_style == 'Stamen Terrain':
        mapbox_style = "stamen-terrain"
    elif map_style == 'USGS Imagery':
        layers.append({
            "sourcetype": "raster",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"],
            "below": "traces"
        })
    elif map_style == 'Canada Weather Radar':
        layers.append({
            "sourcetype": "raster",
            "source": [
                "https://geo.weather.gc.ca/geomet/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"
            ],
            "below": "traces"
        })

    return mapbox_style, layers


def map_observations(data: pd.DataFrame, soil_properties: str, map_style: str) -> None:
    '''
    Plot map observations using Plotly and Mapbox.

    Args:
        data (pd.DataFrame): The input data containing geographic and soil information.
        soil_properties (str): Column name representing the soil property to visualize.
        map_style (str): The style of the map (e.g., 'OpenStreetMap', 'CartoDB Positron').
    '''
    zoom, center = 4, {"lat": 42.0285, "lon": -93.85}

    fig = px.scatter_mapbox(
        data, lat="latitude", lon="longitude",
        size="soil_organic_carbon", color=soil_properties,
        hover_data=['soil_id'], color_continuous_scale=px.colors.cyclical.IceFire,
        zoom=zoom, center=center, height=500
    )

    mapbox_style, layers = mstyle_n_layers(map_style)

    if len(layers) == 0:
        fig.update_layout(mapbox_style=mapbox_style)
    else:
        fig.update_layout(mapbox_style=mapbox_style, mapbox_layers=layers)

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig)
    st.write(f"Circle diameter is proportional to observed SOC content.")


def geocode_location(location_input: str) -> tuple[float, float]:
    '''
    Geocode a location using Nominatim.

    Args:
        location_input (str): The input address to geocode.

    Returns:
        tuple: A tuple containing the latitude and longitude.
    '''
    geolocator = Nominatim(user_agent="geoapiExercises")

    try:
        location = geolocator.geocode(location_input, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            raise ValueError("Location not found")
    except GeocoderTimedOut:
        raise TimeoutError("Geocoding service timed out")
    except Exception as e:
        st.error(f"Geocoding service error: {e}")


def map1(data: pd.DataFrame, soil_properties: str, map_style: str, center: tuple[float, float]) -> None:
    '''
    Create a Folium map with soil data.

    Args:
        data (pd.DataFrame): Data containing geographic and soil information.
        soil_properties (str): Column name representing the soil property to visualize.
        map_style (str): The style of the map (e.g., 'open-street-map').
        center (tuple): The center coordinates (latitude, longitude).
    '''
    zoom = 4
    colormap = cm.linear.YlOrRd_09.scale(data[soil_properties].min(), data[soil_properties].max())

    folium_map = folium.Map(location=center, zoom_start=zoom, tiles=map_style)

    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['soil_organic_carbon'] / 20,
            color=colormap(row[soil_properties]),
            fill=True,
            fill_color=colormap(row[soil_properties]),
            fill_opacity=0.7,
            popup=f"Soil ID: {row['soil_id']}, {soil_properties}: {row[soil_properties]:.2f}, SOC: {row['soil_organic_carbon']:.2f}"
        ).add_to(folium_map)

    colormap.add_to(folium_map)
    folium_map.save("map.html")
    st.components.v1.html(open("map.html", 'r').read(), height=500)
    st.write("Circle diameter is proportional to SOC content.")


def create_map_with_geotiff(geotiff_path: str) -> tuple[Image.Image, rasterio.coords.BoundingBox]:
    '''
    Create a map from a GeoTIFF file.

    Args:
        geotiff_path (str): The path to the GeoTIFF file.

    Returns:
        tuple: A tuple containing the generated image and its bounds.
    '''
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        data = np.where(np.isinf(data), np.nan, data)

        valid_data = data[~np.isnan(data)]
        data = (data - valid_data.min()) / (valid_data.max() - valid_data.min())
        colored_data = plt.get_cmap('plasma')(data)
        colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)

        img = Image.fromarray(colored_data)
        return img, src.bounds


def get_location(address: str) -> tuple[float, float]:
    '''
    Get the coordinates (latitude and longitude) of a given address using Mapbox API.

    Args:
        address (str): The input address.

    Returns:
        tuple: A tuple containing the latitude and longitude.
    '''
    API_KEY = st.secrets["API_KEY"]
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data['features']:
            location = data['features'][0]['geometry']['coordinates']
            return location[1], location[0]

    return None, None


def get_precipitation_value(lat: float, lon: float, geotiff_path: str) -> float:
    '''
    Get the precipitation value for given coordinates from a GeoTIFF file.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        geotiff_path (str): The path to the GeoTIFF file.

    Returns:
        float: The precipitation value at the specified location.
    '''
    with rasterio.open(geotiff_path) as src:
        row, col = src.index(lon, lat)
        value = src.read(1)[row, col]
        return value


def map_layers(img: Image.Image, bounds: rasterio.coords.BoundingBox) -> None:
    '''
    Display a map with image layers.

    Args:
        img (Image.Image): The image to overlay on the map.
        bounds (rasterio.coords.BoundingBox): The bounds of the image.
    '''
    st.markdown("""
        <div style="text-align: center; color: black;">
            <h2>Soil Organic Carbon Prediction Tool</h2>
            <p>Currently this tool allows for browsing current-year soil organic carbon data from 2018. Ultimately, it will help scientists understand the changes in soil organic carbon stocks.</p>
        </div>
        """, unsafe_allow_html=True)

    saved_locations = {
        "Arboretum, Madison WI": [43.049704, -89.415023],
        "W14453 WI-21, Coloma, WI": [44.021722, -89.596748],
        "38W8+5GF Boscobel, Wisconsin": [43.095426, -90.683693],
        "J67M+4MM Wisconsin Dells, Wisconsin": [43.613145, -89.766170]
    }

    folium_map = folium.Map(location=[42.0285, -93.85], zoom_start=4)
    folium.Marker([42.0285, -93.85]).add_to(folium_map)

    if img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        img_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_str}",
            bounds=img_bounds,
            opacity=0.5,
            name="GeoTIFF Overlay"
        ).add_to(folium_map)

    folium.LayerControl().add_to(folium_map)
    st_folium(folium_map)