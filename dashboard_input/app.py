import plotly.express as px
import geopandas as gpd
from zipfile import ZipFile
from io import BytesIO
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller
import numpy as np
import ee
import streamlit as st
import geemap.foliumap as geemap

import ee

try:
    ee.Initialize()
except Exception as e: # if the initialization didn't work, web authenticate first
    ee.Authenticate()
    ee.Initialize(project='ee-moros2')




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



def filter_by_shp(df1,shapefile, label):

    """
    df1 is the csv that contains the complete map
    shape file
    """
    # Convert DataFrame to GeoDataFrame
    if label:
        gdf = gpd.GeoDataFrame(
            df1,
            geometry=gpd.points_from_xy(df1.longitude, df1.latitude)
        )

        # shapefile
        roi = gpd.read_file(shapefile)

        # Ensure the coordinate reference system (CRS) matches
        gdf.set_crs(epsg=4326, inplace=True)  # assuming your points are in WGS84
        roi = roi.to_crs(epsg=4326)

        # Perform spatial join to filter points within Wisconsin
        gdf_roi = gpd.sjoin(gdf, roi, op='within')
        gdf_roi = gdf_roi.drop(columns='geometry')

        # Convert back to a regular DataFrame
        df_roi = pd.DataFrame(gdf_roi)

        return df_roi
    else:
        return None

def read_shapefile_from_upload(uploaded_file, df1):
    f_name = gpd.read_file(uploaded_file)

    gdf = gpd.GeoDataFrame(
        df1,
        geometry=gpd.points_from_xy(df1.longitude, df1.latitude)
    )
    # Ensure the coordinate reference system (CRS) matches
    gdf.set_crs(epsg=4326, inplace=True)  # assuming your points are in WGS84
    f_name = f_name.to_crs(epsg=4326)

    # Perform spatial join to filter points within Wisconsin
    gdf_f_name = gpd.sjoin(gdf, f_name, op='within')

    # Drop the geometry column if you don't need it
    gdf_f_name = gdf_f_name.drop(columns='geometry')

    # Convert back to a regular DataFrame if needed
    new_df = pd.DataFrame(gdf_f_name)
    return new_df



def map(data,soil_properties):
    zoom, center = 2.8, {"lat": 39.50, "lon": -98.35}
    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude",
                            size="soil_organic_carbon", color=soil_properties,
                            color_continuous_scale=px.colors.sequential.Viridis_r,
                            zoom=zoom, center=center, height=500)
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ]
            },
            {
                "sourcetype": "raster",
                "sourceattribution": "Government of Canada",
                "source": ["https://geo.weather.gc.ca/geomet/?"
                           "SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857"
                           "&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"],
            }
        ])
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig)
    st.write(f" Circle diameter is proportional to SOC content.")

def calculate_annual_confidence_interval(data, var):
    # GROUP -- ANNUAL
    annual_data = data.resample('Y').mean()
    #st.dataframe(annual_data)
    # ARIMA
    model = sm.tsa.ARIMA(annual_data[var], order=(1, 0, 0))
    results = model.fit()

    # Predict and CI
    forecast = results.get_forecast(steps=2)
    forecast_df = forecast.summary_frame(alpha=0.05)
    #st.dataframe(forecast_df)

    forecast_df['year'] = annual_data.index
    forecast_df = forecast_df.set_index('year')

    return annual_data, forecast_df

def plot_time_series(annual_data, forecast_df, var):
    """Time series plot and CI."""
    fig = go.Figure()

    # Data
    fig.add_trace(go.Scatter(x=annual_data.index, y=annual_data[var], mode='lines', name='Observations'))

    # Predict
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Prediction'))

    # CI
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean_ci_upper'], mode='lines', name='Upper Boundary', line=dict(width=0)))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean_ci_lower'], mode='lines', fill='tonexty', name='Lower Boundary', line=dict(width=0)))

    fig.update_layout(title=f"Time Series Analysis and CI on {var.replace('_',' ').capitalize()}", xaxis_title='Year', yaxis_title=var)
    return fig

def convert_int_columns(df):
  """
  Automatically converts integer-like column names to strings prefixed with "year_".

  Args:
      df (pandas.DataFrame): The DataFrame with potentially integer-like column names.

  Returns:
      pandas.DataFrame: The DataFrame with converted column names (strings).
  """

  # Filter for integer-like columns
  int_like_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(col)]

  # Convert integer-like column names to strings with "year_" prefix
  new_cols = [str(col) for col in int_like_cols]

  # Create a dictionary for renaming
  mapping = dict(zip(int_like_cols, new_cols))

  # Rename columns using the dictionary
  df = df.rename(columns=mapping)

  return df


def tseries_procedure(data, var):
    dataframe = pd.DataFrame()
    for i in range(1, 0, -1):
        dataframe['t-' + str(i)] = data[var].shift(i)
    final_data = pd.concat([data, dataframe], axis=1)
    final_data.dropna(inplace=True)

    dummy = pd.get_dummies(final_data['year'])
    final_data = pd.concat([final_data, dummy], axis=1)

    finaldf = final_data.drop(['year'], axis=1)
    finaldf = finaldf.reset_index(drop=True)
    test_length = 6
    end_point = len(finaldf)
    x = end_point - test_length
    finaldf_train = finaldf.loc[:x - 1, :]
    finaldf_test = finaldf.loc[x:, :]
    finaldf_test_x = finaldf_test.loc[:, finaldf_test.columns != var]
    finaldf_test_y = finaldf_test[var]
    finaldf_train_x = finaldf_train.loc[:, finaldf_train.columns != var]
    finaldf_train_y = finaldf_train[var]
    st.write("Starting model train..")
    rfr = RandomForestRegressor(n_estimators=100, random_state=1)
    st.write(finaldf_train_x)  # Optional: Display model info
    # Apply the conversion function
    finaldf_train_x.columns = finaldf_train_x.columns.astype(str)
    #finaldf_train_x = convert_int_columns(finaldf_train_x.copy())  # Operate on a copy to avoid modifying original data
    #finaldf_train_y = convert_int_columns(finaldf_train_y.copy())  # Operate on a copy to avoid modifying original data

    # Perform feature selection with RFE
    rfe = RFE(rfr, n_features_to_select=4)  # Only two arguments
    rfe_fit = rfe.fit(finaldf_train_x, finaldf_train_y)

    # Get the features selected by RFE
    selected_features = rfe_fit.support_

    # Train the model on the selected features
    finaldf_train_x_reduced = finaldf_train_x[selected_features]  # Select features
    rfr.fit(finaldf_train_x_reduced, finaldf_train_y)

    # Use the trained model for prediction
    y_pred = rfr.predict(finaldf_test_x[:, selected_features])  # Select features for test data

    return final_data

def histogram_var(data, var):
    fig = px.histogram(data, x=var, title=f"{var.replace('_',' ').replace('norm','').capitalize()} - Histogram")
    st.plotly_chart(fig)

@st.cache_data
def get_geom_data(category):
    #Checkout https://github.com/opengeos/streamlit-geospatial/blob/master/pages/10_🌍_Earth_Engine_Datasets.py

    prefix = (
        "https://raw.githubusercontent.com/giswqs/streamlit-geospatial/master/data/"
    )
    links = {
        "national": prefix + "us_nation.geojson",
        "state": prefix + "us_states.geojson",
        "county": prefix + "us_counties.geojson",
        "metro": prefix + "us_metro_areas.geojson",
        "zip": "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_zcta510_500k.zip",
    }

    if category.lower() == "zip":
        r = requests.get(links[category])
        out_zip = os.path.join(DOWNLOADS_PATH, "cb_2018_us_zcta510_500k.zip")
        with open(out_zip, "wb") as code:
            code.write(r.content)
        zip_ref = zipfile.ZipFile(out_zip, "r")
        zip_ref.extractall(DOWNLOADS_PATH)
        gdf = gpd.read_file(out_zip.replace("zip", "shp"))
    else:
        gdf = gpd.read_file(links[category])
    return gdf



st.sidebar.info(
    """
    - Web App URL: <>
    - GitHub repository: <>
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Pfr JH: <https://>
    [GitHub](https://github.com/) 
    """
)


def nlcd():

    # st.header("National Land Cover Database (NLCD)")

    row1_col1, row1_col2 = st.columns([3, 1])
    width = 950
    height = 600

    Map = geemap.Map(center=[40, -100], zoom=4)

    # Select the seven NLCD epoches after 2000.
    years = ["2001", "2004", "2006", "2008", "2011", "2013", "2016", "2019"]

    # Get an NLCD image by year.
    def getNLCD(year):
        # Import the NLCD collection.
        dataset = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")

        # Filter the collection by year.
        nlcd = dataset.filter(ee.Filter.eq("system:index", year)).first()

        # Select the land cover band.
        landcover = nlcd.select("landcover")
        return landcover

    with row1_col2:
        selected_year = st.multiselect("Select a year", years)
        add_legend = st.checkbox("Show legend")

    if selected_year:
        for year in selected_year:
            Map.addLayer(getNLCD(year), {}, "NLCD " + year)

        if add_legend:
            Map.add_legend(
                legend_title="NLCD Land Cover Classification", builtin_legend="NLCD"
            )
        with row1_col1:
            Map.to_streamlit(width=width, height=height)

    else:
        with row1_col1:
            Map.to_streamlit(width=width, height=height)


def dem_p(image_dir):
    dem = ee.Image(image_dir).select('elevation')
    # Calculate slope, aspect, and hillshade.
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem)
    terrain = ee.Terrain.products(dem)
    Map = geemap.Map(center=[40, -100], zoom=4)
    Map.addLayer(dem, {'min': 0, 'max': 3000, 'palette': ['blue', 'green', 'red']}, 'DEM')
    Map.addLayer(slope, {'min': 0, 'max': 60, 'palette': ['00FFFF', '0000FF']}, 'Slope')
    Map.addLayer(aspect, {'min': 0, 'max': 360, 'palette': ['red', 'orange', 'yellow', 'green', 'cyan', 'blue']},
                 'Aspect')
    Map.addLayer(terrain.select('hillshade'), {}, 'Hillshade')
    return Map



def search_data():

    # st.header("Search Earth Engine Data Catalog")

    Map = geemap.Map()

    if "ee_assets" not in st.session_state:
        st.session_state["ee_assets"] = None
    if "asset_titles" not in st.session_state:
        st.session_state["asset_titles"] = None

    col1, col2 = st.columns([2, 1])
    image_choice = ('NASA/ASTER_GED/AG100_003','USGS/3DEP/10m', 'CGIAR/SRTM90_V4')

    dataset = None
    with col2:
        keyword = st.text_input(
            "Enter a keyword to search (e.g., elevation)", "")
        if keyword:
            ee_assets = geemap.search_ee_data(keyword)
            asset_titles = [x["title"] for x in ee_assets]
            asset_types = [x["type"] for x in ee_assets]

            translate = {
                "image_collection": "ee.ImageCollection('",
                "image": "ee.Image('",
                "table": "ee.FeatureCollection('",
                "table_collection": "ee.FeatureCollection('",
            }

            dataset = st.selectbox("Select a dataset", sorted(asset_titles))
            if len(ee_assets) > 0:
                st.session_state["ee_assets"] = ee_assets
                st.session_state["asset_titles"] = asset_titles

            if dataset is not None:
                with st.expander("Show dataset details", True):
                    index = asset_titles.index(dataset)

                    html = geemap.ee_data_html(
                        st.session_state["ee_assets"][index])
                    html = html.replace("\n", "")
                    st.markdown(html, True)

                ee_id = ee_assets[index]["id"]
                uid = ee_assets[index]["uid"]
                st.markdown(f"""**Earth Engine Snippet:** `{ee_id}`""")
                ee_asset = f"{translate[asset_types[index]]}{ee_id}')"
                vis_params = st.text_input(
                    "Enter visualization parameters as a dictionary", {}
                )
                layer_name = st.text_input("Enter a layer name", uid)
                button = st.button("Add dataset to map")
                if button:
                    vis = {}
                    try:
                        if vis_params.strip() == "":
                            # st.error("Please enter visualization parameters")
                            vis_params = "{}"
                        vis = eval(vis_params)
                        if not isinstance(vis, dict):
                            st.error(
                                "Visualization parameters must be a dictionary")
                        try:
                            Map.addLayer(eval(ee_asset), vis, layer_name)
                        except Exception as e:
                            st.error(f"Error adding layer: {e}")
                    except Exception as e:
                        st.error(f"Invalid visualization parameters: {e}")

            with col1:
                imchoice=st.radio("Pre-charched Images ", image_choice, key=2)
                Map = dem_p(imchoice)
                Map.to_streamlit(width=800, height=600)

        else:
            with col1:
                imchoice = st.radio("Image ", image_choice, key=2)
                Map = dem_p(imchoice)
                Map.to_streamlit(width=800, height=600)


def app():
    apps = [#"Earth Engine - Search Earth Engine Data Catalog",
            #"National Land Cover Database (NLCD)",
            "SOC from RF"]

    selected_app = st.selectbox("Select an app", apps)


    if selected_app == "Earth Engine - National Land Cover Database (NLCD)":
        nlcd()
    elif selected_app == "SOC from RF":
        main()
    elif selected_app == "Earth Engine - Search Earth Engine Data Catalog":
        search_data()


def main():
    st.title("Soil Organic Carbon and other soil properties")

    # Load data
    data = pd.read_csv('data/merged_CONUS_dem.csv')
    data['date'] = pd.to_datetime(data['year'], format='%Y')
    data['soil_organic_carbon_stock'] = data['soil_organic_carbon']*data['depth_cm']*1
    data = data.set_index('date')

    # Streamlit UI
    uploaded_file = st.sidebar.file_uploader("Upload AOI *.shp file", type="shp")
    if uploaded_file is not None:
        data = read_shapefile_from_upload(uploaded_file, data)

    soil_properties = st.sidebar.radio("Soil Properties",
                                       ('soil_organic_carbon', 'dem', 'slope_norm', 'aspect_rad',
                                        'hillshade'), key=2)

    depth_c = st.sidebar.radio("Depth (cm)", ('0-5','5-15','15-30','30-60','60-100','100-200'))
    depth = depth_c.split('-')[-1]
    data = data[data['depth_cm']==int(depth)]
    map(data, soil_properties)
    histogram_var(data, soil_properties)


if __name__ == "__main__":
    app()
