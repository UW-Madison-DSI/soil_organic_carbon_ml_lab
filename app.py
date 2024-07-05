import os.path
import plotly.express as px
import geopandas as gpd
import plotly.graph_objs as go
import os
import altair as alt
import pandas as pd
import streamlit as st
#import sklearn
import os
import zipfile
import joblib
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

import plotly.graph_objects as go


st.markdown(
    """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            color: #333333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #007bff;
        }
    </style>
    """,
    unsafe_allow_html=True
)




HERE = os.path.dirname(os.path.abspath(__file__))

st.title("Soil Dynamic Properties.")
DATA = os.path.join(HERE, "data/final_conus.csv")


@st.cache_data
def load_data():
    dta=pd.read_csv("./data/final_conus.csv")
    return dta

states_list=['ALL','ALABAMA',
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


features = ['label','depth_cm', 'min_temperature', 'max_temperature', 'total_precipitation',
       'land_cover', 'land_use', 'clay_mean', 'silt_mean', 'sand_mean',
       'bd_mean', 'dem', 'slope', 'aspect', 'hillshade', 'om_mean', 'ph_mean']
depth_c = st.sidebar.radio("Depth (cm)", ('0-5', '5-15', '15-30', '30-60', '60-100', '100-200', 'ALL'), key=2)

# Dictionary to store loaded models for reuse
loaded_models = {}

def trend_time_series(dframe, depth_c):

    if 'ALL' in depth_c:
        df = dframe
    else:
        df = dframe[dframe['depth_cm']==int(depth_c.split('-')[-1])]

    try:

        df = df.groupby(by='year').agg({'soil_organic_carbon': 'median'})
        df = df.reset_index()

        #sort_index(inplace=True)
        # get the values
        soc_cycle, soc_trend = hpfilter(df['soil_organic_carbon'], lamb=1200)


        df['DESAdd'] = ExponentialSmoothing(df['soil_organic_carbon'], trend='add').fit().fittedvalues.shift(-1)
        df['EWMA3'] = df['soil_organic_carbon'].ewm(alpha=2/3, adjust=False).mean()
        df['SES3'] = SimpleExpSmoothing(df['soil_organic_carbon']).fit(smoothing_level=.1,
                                                         optimized=False).fittedvalues.shift(-1)

        #st.write(soc_cycle)
        fig = go.Figure()

        # Add Sales line
        fig.add_trace(go.Scatter(x=df['year'], y=df['soil_organic_carbon'], mode='lines+markers', name='SOC_observations'))
        # Add Trend line
        #fig.add_trace(go.Scatter(x=df['year'], y=df['DESAdd'], mode='lines', name='SOC_frcast_DESAdd'))
        fig.add_trace(go.Scatter(x=df['year'], y=df['EWMA3'], mode='lines', name='SOC_frcast_EWMA3'))
        fig.add_trace(go.Scatter(x=df['year'], y=df['SES3'], mode='lines', name='SOC_frcast_SES3'))
        # Update layout for autoscaling x-axis
        fig.update_layout(
            title=f'Time series forecasting and observations of SOC over the Time at {depth_c} cm depth',
            xaxis_title='year',
            yaxis_title='Soil Organic Carbon',
            xaxis=dict(
                rangeslider=dict(visible=True),
                autorange=True
            )
        )
        # Display the plotly plot in Streamlit
        st.plotly_chart(fig)
    except Exception as e:
        st.error(e)

def load_model(model_name):
    """
    Loads and returns the machine learning model based on the model name.

    Args:
        model_name (str): The name of the model to load (e.g., 'XGB', 'GBM', 'RF').

    Returns:
        object: The loaded machine learning model.

    Raises:
        ValueError: If the specified model name is not supported.
        FileNotFoundError: If the model file is not found at the expected path.
    """

    if model_name == 'XGB':
        model = joblib.load('models/xgb_model.joblib')

    elif model_name == 'GBM':
        model = joblib.load('models/gbm_model.joblib')

    elif model_name == 'RF':
        model = joblib.load('models/random_forest_model.joblib')

    else:
        raise ValueError("Model not supported")

    return model

def data_process():
    data_load_state = st.text("Loading data...")
    data = load_data()
    data_load_state.text("")
    data['soil_organic_carbon_stocks'] = data['soil_organic_carbon'] * data['depth_cm'] * data['bd_mean']


    #st.write(features1)
    data['date'] = pd.to_datetime(data['year'], format='%Y')

    data = data.set_index('date')
    #st.write(len(data))
    earliest_year = data["year"].min()
    latest_year = data["year"].max()

    min_year, max_year = st.sidebar.slider(
        "Year Range",
        min_value=int(earliest_year),
        max_value=2020,
        value=[int(earliest_year), int(latest_year)],
    )

    filtered_data = data[data["year"] >= min_year-1]
    filtered_data = filtered_data[filtered_data["year"] <= max_year]
    return filtered_data

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




def map(data,soil_properties, map_style):
    zoom, center = 2.8, {"lat": 39.50, "lon": -98.35}


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

    st.write(f" Circle diameter is proportional to SOC content.")

def histogram_var(data, var, label):
    if label is True:
        plot_model_comparisons(data, features)


    fig = px.histogram(data, x=var, title=f"{var.replace('_',' ').replace('norm','').replace('mean','').replace('om','Organic matter').replace('bd','Bulk density').capitalize()} - Distribution")
    st.plotly_chart(fig)
    #st.write(data[var].describe())

def logic(states_list,features1,depth_c, label):
    filtered_data=data_process()
    map_style = st.sidebar.selectbox(
        'Select the map style:',
        ('OpenStreetMap',  # 'CartoDB Positron', 'Stamen Terrain',
         'USGS Imagery'  # , 'Canada Weather Radar'
         )
    )
    select_state = st.sidebar.selectbox("State", states_list, key=4)
    states = gpd.read_file('data/states_shape/States_shapefile.shp')
    if select_state is not None and select_state != 'ALL':
        gdf = gpd.GeoDataFrame(filtered_data, geometry=gpd.points_from_xy(filtered_data.longitude, filtered_data.latitude))
        state_gdf = states[states['State_Name'] == select_state]
        # Perform spatial join
        filtered_data0 = gpd.sjoin(gdf, state_gdf, how="inner")
    else:
        filtered_data0 = filtered_data

    soil_properties_c = st.sidebar.radio("Soil Properties",
                                       (features1
                                        #'soil_organic_carbon_predictions','soil_organic_carbon_stocks'
                                       ), key=3)
    soil_properties = soil_properties_c.replace(" (observed)","")
    terraincls = ['bd_mean','clay_mean', 'om_mean', 'ph_mean', 'sand_mean', 'silt_mean']
    #depth_c = st.radio("Depth (cm)", ('0-5', '5-15', '15-30', '30-60', '60-100', '100-200', 'ALL'))
    if soil_properties in terraincls:
        depth_c1 = depth_c
        #st.sidebar.radio("Depth (cm)", ('0-5', '5-15', '15-30', '30-60', '60-100', '100-200', 'ALL'), key=1)
        filtered_data1 = filtered_data0
        if 'ALL' not in depth_c1:
            depth = depth_c1.split('-')[-1]
            filtered_data1 = filtered_data0[filtered_data0['depth_cm'] == int(depth)]
    else:
        filtered_data1 = filtered_data0[filtered_data0['depth_cm'] == 5]

    #st.write("Soil Profiles: ", len(filtered_data1['soil_id'].drop_duplicates()))
    st.write("Total sample data: ", len(filtered_data1))

    map(filtered_data1, soil_properties, map_style)
    trend_time_series(filtered_data0, depth_c)

    histogram_var(filtered_data1, soil_properties, label)

if __name__ == "__main__":
    features1 = ['land_cover', 'land_cover_class', 'land_use', 'land_use_class',
                 'total_precipitation', 'min_temperature', 'mean_temperature', 'max_temperature',
                 'dem', 'slope', 'aspect', 'hillshade', 'bd_mean', 'clay_mean', 'om_mean', 'ph_mean',
                 'sand_mean', 'silt_mean', 'soil_organic_carbon', 'soil_organic_carbon_stocks', 'label']
    logic(states_list, features1, depth_c, False)

