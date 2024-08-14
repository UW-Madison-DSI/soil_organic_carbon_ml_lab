import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import requests

from src.app_functions.functions import histogram_var
from src.app_functions.map_layers import *


@st.cache_data
def load_data_conus():
    """
    Load CONUS data from a Parquet file.
    Returns:
        DataFrame: The loaded dataset.
    """
    return pd.read_parquet('data/sample_soc_observations/final_conus_v2.parquet', engine='pyarrow')


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points on the earth.
    Args:
        lat1, lon1: Latitude and longitude of the first point.
        lat2, lon2: Latitude and longitude of the second point.
    Returns:
        float: Distance between the two points in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c  # Radius of Earth in kilometers


def filter_within_radius(df, lat_point, lon_point, radius_km):
    """
    Filter DataFrame rows that are within a specified radius from a point.
    Args:
        df (DataFrame): The input DataFrame with latitude and longitude columns.
        lat_point (float): Latitude of the center point.
        lon_point (float): Longitude of the center point.
        radius_km (float): Radius in kilometers.
    Returns:
        DataFrame: Filtered DataFrame with rows within the radius.
    """
    df['distance'] = df.apply(lambda row: haversine(lat_point, lon_point, row['latitude'], row['longitude']), axis=1)
    return df[df['distance'] <= radius_km]


def map_plot(data):
    """
    Plot data on a hexbin map using Plotly and Streamlit.
    Args:
        data (DataFrame): Data to plot with latitude, longitude, and other required columns.
    """
    fig = ff.create_hexbin_mapbox(
        data_frame=data,
        lat="latitude",
        lon="longitude",
        nx_hexagon=2,
        opacity=0.3,
        labels={"color": "Land Cover"},
        color="land_use",
        agg_func=np.mean,
        color_continuous_scale="Viridis",
        show_original_data=True,
        original_data_marker=dict(size=4, opacity=0.6, color="black"),
    )
    fig.update_traces(
        hovertemplate="<b>Land Cover</b>: %{customdata[0]}<br><b>Land Use</b>: %{customdata[1]}<br><extra></extra>",
        customdata=data[['land_cover_class', 'land_use_class']].values
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)


def soc_prediction(lat, lon, km_filter):
    """
    Predict SOC based on land use and land cover data.
    Args:
        lat (float): Latitude for the center of the analysis.
        lon (float): Longitude for the center of the analysis.
        km_filter (float): Radius in kilometers to filter the data.
    """
    year = st.selectbox("Select the year", [1990, 2018], key=101)
    df_path = f'data/grid{year}/lulc_{year}_wi.parquet'
    df = pd.read_parquet(df_path)

    if lat is None:
        zoom, center = 6, {"lat": 44.723802, "lon": -89.961530}
    else:
        zoom, center = 10.5, {"lat": lat, "lon": lon}

    if km_filter:
        df = filter_within_radius(df, lat, lon, km_filter)
        map_plot(df)
    else:
        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color='land_use_class',
            hover_data=['land_use_class', 'land_cover_class'],
            color_continuous_scale=px.colors.sequential.Inferno_r,
            zoom=zoom,
            center=center,
            height=500,
            mapbox_style="open-street-map"
        )
        fig.update_layout(mapbox=dict(style="open-street-map", center=center, zoom=zoom))
        st.plotly_chart(fig)


def map_layers_prediction():
    """
    Display the SOC prediction tool interface and handle user input.
    """
    st.markdown("""
        <div style="text-align: center; color: black;">
            <h2>Soil Organic Carbon Prediction Tool</h2>
            <p>Currently, this tool allows for land use and land cover data in 2018. 
            Ultimately, it will help scientists understand the changes in soil organic carbon stocks.</p>
        </div>
        """, unsafe_allow_html=True)

    lat = 44.723802
    lon = -89.961530

    try:
        input_option = st.sidebar.checkbox("Input address")
        if input_option:
            address = st.sidebar.text_input("Please introduce an address of interest:", value="Arboretum, Madison WI")
            locations = {
                'Arboretum': (43.049704, -89.415023),
                'W14453 WI-21': (44.021722, -89.596748),
                '38W8+5GF Boscobel': (43.095426, -90.683693),
                'J67M+4MM Wisconsin Dells': (43.613145, -89.766170),
            }
            lat, lon = locations.get(address, get_location(address))
        else:
            lat = st.sidebar.number_input("Latitude:", value=43.064)
            lon = st.sidebar.number_input("Longitude:", value=-89.407)
    except Exception as e:
        st.error(f"No address >>> {e}")

    km_input = st.sidebar.radio("Ratio around the location (km)", [None, 2, 4], key='visibility')
    soc_prediction(lat, lon, km_input)