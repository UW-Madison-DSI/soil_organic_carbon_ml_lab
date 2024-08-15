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
    color_map = {
        "Forest": "#33FF57",  # Example color for Class A
        "Other": "#EF553B",  # Example color for Class B
        "Non-Forest Wetland": "#00CC96",
        "Developed": "#FF5733",
        "Agriculture": "#F7DC6F",
        "Rageland or Pasture": "#F1948A",# Example color for Class C
        # Add more classes and colors as needed
    }

    fig = ff.create_hexbin_mapbox(
        data_frame=data,
        lat="latitude",
        lon="longitude",
        nx_hexagon=2,
        opacity=0.3,
        #labels={"color": "Land Cover"},
        color="land_use_class",
        #agg_func=np.mean,
        color_continuous_scale=color_map,
        show_original_data=True,
        #original_data_marker=dict(size=4, opacity=0.6, color="black"),
    )
    fig.update_traces(
        hovertemplate="<b>Land Cover</b>: %{customdata[0]}<br><b>Land Use</b>: %{customdata[1]}<br><extra></extra>",
        customdata=data[['land_cover_class', 'land_use_class']].values
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)


def soc_prediction(df, lat, lon, km_filter):
    """
    Predict SOC based on land use and land cover data.
    Args:
        lat (float): Latitude for the center of the analysis.
        lon (float): Longitude for the center of the analysis.
        km_filter (float): Radius in kilometers to filter the data.
    """
    #year = st.selectbox("Select the year", [1990, 2018], key=101)
    #df_path = f'data/grid{year}/lulc_{year}_wi.parquet'
    #df = pd.read_parquet(df_path)

    if lat is None:
        zoom, center = 4, {"lat": 44.723802, "lon": -89.961530}
    else:
        zoom, center = 8, {"lat": lat, "lon": lon}

    if df is None:
        pass
        #df = filter_within_radius(df, lat, lon, km_filter)

        #map_plot(df)
    else:
        color_map = {
            "Forest": "#228B22",  # Bright green
            "Other": "#90d5ff",  # Red-orange
            "Non-Forest Wetland": "#00CC96",  # Greenish-blue
            "Developed": "#FF5733",  # Bright red-orange
            "Agriculture": "#f0f921",  # Yellow
            "Rangeland or Pasture": "#F1948A",  # Light coral
            "Water": "#007FFF",  # Blue
            "Barren Land": "#E0FFFF",  # Light cyan
            # Add more classes and colors as needed
        }

        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color='land_use_class',
            hover_data=['land_use_class', 'land_cover_class'],
            color_discrete_map=color_map,  # Use color_discrete_map instead of color_continuous_scale
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

    km_filter = st.sidebar.radio("Ratio around the location (km)", [None, 2, 4], key='visibility')

    yr1 = 1990
    df_path1 = f'data/grid{yr1}/lulc_{yr1}_wi.parquet'
    df1 = pd.read_parquet(df_path1)
    year2 = 2018
    df_path2 = f'data/grid{year2}/lulc_{year2}_wi.parquet'
    df2 = pd.read_parquet(df_path2)



    cl1, cl2 = st.columns([2, 2])
    if km_filter:
        #with cl1:
        st.write("## 1990")
        df1 = filter_within_radius(df1, lat, lon, km_filter)
        map_plot(df1)
        #with cl2:
        st.write("## 2018")
        df2 = filter_within_radius(df2, lat, lon, km_filter)
        map_plot(df2)
    else:
        #with cl1:
        st.write("## 1990")
        soc_prediction(df1, lat, lon, km_filter)
        #with cl2:
        st.write("## 2018")
        soc_prediction(df2, lat, lon, km_filter)