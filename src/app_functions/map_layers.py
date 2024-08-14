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

from src.app_functions.functions import *
import requests
def mstyle_n_layers(map_style):
    layers = []
    mapbox_style = "open-street-map"
    if map_style == 'OpenStreetMap':
        mapbox_style = "open-street-map"
    elif map_style == 'CartoDB Positron':
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
                "https://geo.weather.gc.ca/geomet/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"],
            "below": "traces"
        })
    return mapbox_style, layers

def map_observations(data,soil_properties, map_style):
    zoom, center = 2.7, {"lat": 39.50, "lon": -98.35}


    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude",
                            size="soil_organic_carbon", color=soil_properties,
                            hover_data=['soil_id'],
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            zoom=zoom, center=center, height=500)

    mapbox_style, layers = mstyle_n_layers(map_style)
    if len(layers)==0:
        fig.update_layout(
            mapbox_style="open-street-map"
        )
    else:
        fig.update_layout(
            mapbox_style=mapbox_style,
            mapbox_layers=layers
        )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig)

    st.write(f" Circle diameter is proportional to observed SOC content.")

def geocode_location(location_input):
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

def map1(data, soil_properties, map_style,center):

    # Set initial zoom and center
    zoom = 5
    colormap = cm.linear.YlOrRd_09.scale(data[soil_properties].min(), data[soil_properties].max())

    # Initialize Folium map
    folium_map = folium.Map(location=center, zoom_start=zoom, tiles=map_style)

    # Add data points to the map
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['soil_organic_carbon']/20,  # Adjust radius as needed
            color=colormap(row[soil_properties]),
            fill=True,
            fill_color=colormap(row[soil_properties]),
            fill_opacity=0.7,
            popup=f"Soil ID: {row['soil_id']}, {soil_properties}: {row[soil_properties]:.2f}, soc: {row['soil_organic_carbon']:.2f}"
        ).add_to(folium_map)

    # Customize the map style and layers
    if map_style == 'open-street-map':
        folium.TileLayer('Open Street Map').add_to(folium_map)
    else:
        folium.TileLayer(map_style).add_to(folium_map)

    # Display the map in Streamlit
    colormap.add_to(folium_map)
    folium_map.save("map.html")
    st.components.v1.html(open("map.html", 'r').read(), height=500)

    st.write("Circle diameter is proportional to SOC content.")

def create_map_with_geotiff(geotiff_path):
    # Read the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        # Get the bounds of the GeoTIFF
        bounds = src.bounds
        srcdata = src.read()#[0]  # read raster vals into numpy array

        # Handle different numbers of bands
        if src.count == 1:  # Single band
            data = srcdata[0]

            # Replace -inf values with NaN
            data = np.where(np.isinf(data), np.nan, data)

            # Display data statistics
            # Filter out extremely large negative values (e.g., below a threshold)
            threshold = -1e20
            data = np.where(data < threshold, np.nan, data)

            # Display data statistics after filtering
            # Remove NaN values for normalization
            valid_data = data[~np.isnan(data)]

            # Normalize the data
            data_min = valid_data.min()
            data_max = valid_data.max()
            data = (data - data_min) / (data_max - data_min)
            data = np.nan_to_num(data)  # Replace NaN with 0 after normalization

            # Apply a colormap
            cm = plt.get_cmap('plasma')
            colored_data = cm(data)

            colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)

        elif src.count == 3:  # RGB
            st.write('band 1')
            # Transpose to put bands last
            data = np.transpose(srcdata, (1, 2, 0))
            # Normalize the data
            data = (data - data.min()) / (data.max() - data.min())
            colored_data = (data * 255).astype(np.uint8)
        elif src.count == 4:  # RGBA
            # Transpose to put bands last
            data = np.transpose(srcdata, (1, 2, 0))
            # Use data as is, assuming it's already in 0-255 range
            colored_data = data.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported number of bands: {srcdata.shape[0]}")

        # Create a PIL Image
        img = Image.fromarray(colored_data)
        return img, bounds

def get_location(address):
    API_KEY = st.secrets["API_KEY"]
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['features']:
            location = data['features'][0]['geometry']['coordinates']
            return location[1], location[0]
    return None, None


def get_precipitation_value(lat, lon, geotiff_path):
    with rasterio.open(geotiff_path) as src:
        # Transform geographic coordinates to image coordinates
        row, col = src.index(lon, lat)
        # Read the pixel value at the coordinates
        value = src.read(1)[row, col]
        return value

def map_layers(img, bounds):
    st.markdown("""
        <div style="text-align: center; color: black;">
            <h2>Soil Organic Carbon Prediction Tool</h2>
            <p>Currently this tool allows for browsing current-year soil organic carbon data from 2018. Ultimately, it will help scientists understand the changes in soil organic carbon stocks.</p>
        </div>
        """, unsafe_allow_html=True)

    saved_locations = {"Arboretum, Madison WI":[43.049704, -89.415023],
                       "W14453 WI-21, Coloma, WI 54930":[44.021722, -89.596748],
                       "38W8+5GF Boscobel, Wisconsin":[43.095426, -90.683693],
                       "J67M+4MM Wisconsin Dells, Wisconsin":[43.613145, -89.766170]
                       }

    lat = 43.049704
    lng = -89.415023
    zoom = 3
    try:
        input_option = st.sidebar.checkbox("Input address")

        if input_option:
            address = st.sidebar.text_input("Please introduce an address of interest:", value="Arboretum, Madison WI")

            if 'Alboretum' in address:
                lat, lon = 43.049704, -89.415023
            elif 'W14453 WI-21' in address:
                lat, lon = 44.021722, -89.596748
            elif '38W8+5GF Boscobel' in address:
                lat, lon = 43.095426, -90.683693
            elif 'J67M+4MM Wisconsin Dells' in address:
                lat, lon = 43.613145, -89.766170
            else:
                st.write('here2 ', (saved_locations.keys()))

                lat, lng = get_location(address)

        else:
            lat = st.sidebar.number_input("Latitude:", value=43.064)
            lng = st.sidebar.number_input("Longitude:", value=-89.407)

        zoom=13
    except Exception as e:
        st.error(f"No address >>> {e}")

    value = get_precipitation_value(lat, lng, "data/prcp_1850 copy.tif")
    st.write(f"Precipitation value: {value:.2f}")

    m = folium.Map(location=[lat, lng], zoom_start=zoom)
    folium.Marker([lat, lng]).add_to(m)

    # Add the Esri tile layer
    esri_url = ('https://server.arcgisonline.com/ArcGIS/rest/services/'
                'World_Topo_Map/MapServer/tile/{z}/{y}/{x}')

    attribution = ('Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, '
                   'FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, '
                   'Esri China (Hong Kong), and the GIS User Community')

    TileLayer(
        tiles=esri_url,
        attr=attribution,
        name='Esri World Topo Map',
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
        ),
        name='Carto CDN',
        overlay=False,
        control=True
    ).add_to(m)

    # Add the MapTiler tile layer
    folium.TileLayer(
        tiles="https://api.maptiler.com/maps/topo/{z}/{x}/{y}.png?key="+str(maptilerkey),
        attr='Map data © OpenStreetMap contributors, Imagery © MapTiler',
        name='Map Tiler',
        overlay=False,
        control=True
    ).add_to(m)

    # Convert PIL Image to base64 string
    if img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Add the GeoTIFF as an image overlay
        img_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_str}",
            bounds=img_bounds,
            opacity=0.5,
            name="GeoTIFF Overlay"
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    # JavaScript to get the pixel value on mouse move
    custom_js = """
        <script>
        function getColor(e) {
            var pixel = map.layerPointToContainerPoint(e.layerPoint);
            var canvas = document.querySelector('canvas');
            var context = canvas.getContext('2d');
            var imgd = context.getImageData(pixel.x, pixel.y, 1, 1).data;
            var value = imgd[0];
            alert('Value: ' + value);
        }
        map.on('mousemove', getColor);
        </script>
        """

    # Add the custom JavaScript to the map
    folium.Marker(
        location=[lat, lng],
        icon=folium.DivIcon(html=custom_js)
    ).add_to(m)

    folium_static(m)
    #return m

def map22(geotiff_path):
    # Read the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        dataset = rasterio.open(src, 'r')  # open raster file

        rasdata = dataset.read()[0]  # read raster vals into numpy array

        rasdata_normed = rasdata / rasdata.max() * 10  # normalization to help with color gradient

        # set bounds using wgs84 projection (geotiff is in web mercator)
        dest_crs = 'EPSG:4326'
        left, bottom, right, top = [i for i in dataset.bounds]
        bounds_4326 = warp.transform_bounds(src_crs=dataset.crs, dst_crs=dest_crs, left=left,
                                            bottom=bottom, right=right, top=top)

        # map it out!
        m = folium.Map([43.071079, -89.418544], zoom_start=8)
        folium.raster_layers.ImageOverlay(
            image=rasdata_normed,
            name='sample map',
            opacity=0.5,
            bounds=[[bounds_4326[1], bounds_4326[0]], [bounds_4326[3], bounds_4326[2]]],
            interactive=False,
            cross_origin=False,
            zindex=1,
            colormap=cm.get_cmap('Blues', 10)
        ).add_to(m)
        folium.LayerControl().add_to(m)

        return m
