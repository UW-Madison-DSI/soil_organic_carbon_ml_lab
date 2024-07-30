import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium import CircleMarker
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.express as px
import plotly.graph_objs as go
import altair as alt
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import branca.colormap as cm

import os.path
import rasterio
from rasterio import warp
from rasterio.plot import show
#import sklearn
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import statsmodels.api as sm

from tempfile import NamedTemporaryFile
from scipy.stats import t
import plotly.graph_objects as go
from folium import TileLayer
from streamlit_folium import folium_static
import tempfile
import base64
from PIL import Image
import folium
#from StringIO import io
import io
import pandas as pd


st.markdown(
    """
    <style>
        body {
            font-family: 'Lato', sans-serif;
            background-color: #f0f0f0;
            color: #333333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #48a67b;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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
@st.cache_data
def load_data():
    dta=pd.read_csv("data/final_conus_v2.csv")
    return dta


def load_data_conus():
    # Use an absolute path
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "data", "final_conus_v2.csv")

    # Check the current working directory and file path
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Loading data from: {file_path}")

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None

    # Load the data
    dta = pd.read_csv(file_path)
    return dta

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
            #for state in df_pivot.columns:
            #    models[state] = sm.tsa.SARIMAX(df_pivot[state], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

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
            plt.title('Soil Organic Carbon Time Series with Forecast and Confidence Intervals')
            st.pyplot(plt)

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

def plot_model_comparisons(data, features):
    try:


        # Load your dataset
        # df = pd.read_csv('path_to_your_data.csv')
        X = data.drop('soil_organic_carbon', axis=1)  # replace 'target_column' with the name of your target variable
        y = data['soil_organic_carbon']

        # Split the dataset into train (if needed for further tuning) and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predict on the test set
        rf_model = joblib.load('models/random_forest_model.joblib')
        xgb_model = joblib.load('models/xgb_model.joblib')
        rf_predictions = rf_model.predict(X_test[features])
        xgb_predictions = xgb_model.predict(X_test[features])

        # Calculate RMSE
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

        # Plotting the results
        metrics = [rf_rmse, xgb_rmse]
        model_names = ['Random Forest', 'XGBoost']

        fig = px.bar(x=model_names, y=metrics, title='MSE Performance Comparison of Pre-trained Models on New Data',
                     color_discrete_sequence=['forestgreen', 'darkorange'])  # Create plotly bar chart with colors

        # Assuming you have Streamlit integration (st is a Streamlit object)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(e)

def modeling(model_name, data, features):
    """
    Apply the specified machine learning model to predict soil organic carbon.
    Parameters:
        model_name (str): The name of the model ('SVR', 'GBR', 'RF').
        data (DataFrame): The dataset containing features.
        features (list): The list of feature names used for prediction.
    Returns:
        DataFrame: The input data with an additional column for predictions.
    """
    try:
        model = None#load_model(model_name)

        data1 = data.copy()
        data1['soil_organic_carbon_predictions'] = model.predict(data1[features])
        return data1
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return data


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

def map1(data,soil_properties, map_style):
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

    st.write(f" Circle diameter is proportional to SOC content.")


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

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


def map(data, soil_properties, map_style,center):


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


def histogram_var(data, var, label):
    if label is True:
        plot_model_comparisons(data, features)

    fig = px.histogram(data, x=var, title=f"{var.replace('_',' ').replace('norm','').replace('mean','').replace('om','Organic matter').replace('bd','Bulk density').capitalize()} - Distribution")
    st.plotly_chart(fig)
    #st.write(data[var].describe())


def upload1():
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

import requests
def get_location():
    import streamlit as st
    from shapely.geometry import Point, Polygon
    import geopandas as gpd
    import pandas as pd
    import geopy

    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    street = st.sidebar.text_input("Street", "University Avenue")
    city = st.sidebar.text_input("City", "Madison")
    #province = st.sidebar.text_input("Province", "Madison")
    country = st.sidebar.text_input("Country", "United States of America")

    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street + ", " + city + ", " + country)

    lat = location.latitude
    lon = location.longitude

    return lat, lon

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
            <p>Currently this tool allows for browsing current-year soil organic carbon data. Ultimately, it will help scientists understand the changes in soil organic carbon stocks.</p>
        </div>
        """, unsafe_allow_html=True)

    lat = 44.900771
    lng = -89.5694905
    zoom = 3
    try:
        lat, lng = get_location()
        #st.write(lat, lng)
        zoom=15
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
        tiles="https://api.maptiler.com/maps/topo/{z}/{x}/{y}.png?key=xcdVPbBPNrw13P7FG6Ot",
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

def my_data():
    # pass

    HERE = os.path.dirname(os.path.abspath(__file__))

    st.title("Soil Organic Carbon Tool.")
    data = load_data_conus()
    #pd.read_csv("data/final_conus_v2.csv")
    depth_c = st.sidebar.radio("Depth (cm)", ('0-5', '5-15', '15-30', '30-60', '60-100', '100-200', 'ALL'), key=2)

    uploaded_file = None
    # st.sidebar.file_uploader("Test my own data (CSV file)", type=['csv'])
    label = False
    if uploaded_file is not None:
        label = True
        try:
            # select_ml = st.sidebar.selectbox("Predictive modeling:", ["XGB", #"GBM",
            #                                                          "RF"], key=1011)
            # df = pd.read_csv(uploaded_file)
            # data_frame = df.copy()
            # data = modeling(df1, select_ml, features)
            data = None
            # modeling(select_ml, data_frame, features)

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

        # st.write(features1)
    data['date'] = pd.to_datetime(data['year'], format='%Y')

    data = data.set_index('date')
    # st.write(len(data))
    earliest_year = data["year"].min()
    latest_year = data["year"].max()
    min_year, max_year = st.sidebar.slider(
        "Year Range",
        min_value=int(earliest_year),
        max_value=2020,
        value=[int(earliest_year), int(latest_year)],
    )

    filtered_data = data[data["year"] >= min_year - 1]
    filtered_data = filtered_data[filtered_data["year"] <= max_year]

    try:
        map_style = st.sidebar.selectbox(
            'Select the map style:',
            ('OpenStreetMap',  # 'CartoDB Positron', 'Stamen Terrain',
             'USGS Imagery'  # , 'Canada Weather Radar'
             )
        )
        select_state = st.sidebar.selectbox("State", states_list, key=4)
        states = gpd.read_file('states_shape/States_shapefile.shp')
        if select_state is not None and select_state != 'ALL':
            gdf = gpd.GeoDataFrame(filtered_data,
                                   geometry=gpd.points_from_xy(filtered_data.longitude, filtered_data.latitude))
            state_gdf = states[states['State_Name'] == select_state]
            # Perform spatial join
            filtered_data0 = gpd.sjoin(gdf, state_gdf, how="inner")
        else:
            filtered_data0 = filtered_data

        # st.write(filtered_data0.columns)
        soil_properties_c = st.sidebar.radio("Soil Properties",
                                             (features1
                                              # 'soil_organic_carbon_predictions','soil_organic_carbon_stocks'
                                              ), key=3)
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
        trend_time_series(filtered_data0, depth_c, 'ALL')

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
        if password == "$jhlab2024$":  # Replace with secure password storage
            st.session_state.authenticated = True
            st.experimental_set_query_params(rerun=True)# Rerun the app to reflect the login state
        else:
            st.error("Incorrect password")

def main():
    #st.write("# Soil Organic Carbon Tool! 👋")

    st.sidebar.title("Functionalities")
    sidebar_object = st.sidebar.radio('Please choose', ("SOC Assistant",'My Data'), key=100)

    if sidebar_object == 'SOC Assistant':
        soc_prediction()

    else:
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False

        if st.session_state.authenticated==False:
            login()
        else:
            my_data()

main()