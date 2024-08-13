import streamlit as st
import plotly.express as px
import streamlit as st
import plotly.figure_factory as ff
import requests
from src.app_functions.functions import histogram_var
from src.app_functions.map_layers import *

@st.cache_data
def load_data_conus():
    '''

    :return:
    '''
    dta = pd.read_parquet('data/sample_soc_observations/final_conus_v2.parquet', engine='pyarrow')
    return dta


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r = 6371  # Radius of Earth in kilometers
    return r * c

def filter_within_radius(df, lat_point, lon_point, radius_km):
    '''

    :param df:
    :param lat_point:
    :param lon_point:
    :param radius_km:
    :return:
    '''
    # Calculate distance from the given point
    df['distance'] = df.apply(lambda row: haversine(lat_point, lon_point, row['latitude'], row['longitude']), axis=1)

    # Filter points within the specified radius
    result_df = df[df['distance'] <= radius_km]

    return result_df


def map_plot(data):
    '''

    :param data:
    :return:
    '''
    fig = ff.create_hexbin_mapbox(
        data_frame=data,
        lat="latitude",
        lon="longitude",
        nx_hexagon=2,
        opacity=.3,
        labels={"color": "Land Cover"},
        color="land_use",
        agg_func=np.mean,
        color_continuous_scale="Viridis",
        show_original_data=True,
        original_data_marker=dict(size=4, opacity=0.6, color="black"),

    )
    fig.update_traces(
        hovertemplate="<b>Land Cover</b>: %{customdata[0]}<br>" +
                      "<b>Land Use</b>: %{customdata[1]}<br>" +
                      "<extra></extra>",
        customdata=data[['land_cover_class','land_use_class']].values
    )
    fig.update_layout(mapbox_style="open-street-map")
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def soc_prediction(lat, lon, km_filter):
    '''

    :param lat:
    :param lon:
    :param myzone:
    :return:
    '''

    year = st.selectbox("Select the year",[1990,2018],key=101)

    df = pd.read_parquet('data/grid1990/lulc_1990_wi.parquet')
    if year == 2018:
        df = pd.read_parquet('data/grid2018/lulc_2018_wi.parquet')

    # Define zoom and center based on whether lat is provided
    if lat is None:
        zoom, center = 6, {"lat": 44.723802, "lon": -89.961530}
    else:
        zoom, center = 10.5, {"lat": lat, "lon": lon}

    if km_filter:
        df = filter_within_radius(df, lat, lon, km_filter)
        map_plot(df)

    else:
        # Create the main scatter mapbox plot
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
            mapbox_style="open-street-map"  # Ensure markers are visible
        )

        # Update layout to ensure map settings
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # Use a style that supports markers
                center=dict(lat=center['lat'], lon=center['lon']),
                zoom=zoom
            )
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig)


def map_layers_prediction():
    '''

    :return:
    '''
    st.markdown("""
        <div style="text-align: center; color: black;">
            <h2>Soil Organic Carbon Prediction Tool</h2>
            <p>Currently this tool allows for land use and land cover data in 2018. Ultimately, it will help scientists understand the changes in soil organic carbon stocks.</p>
        </div>
        """, unsafe_allow_html=True)


    lat = 44.723802
    lng = -89.961530
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
                lat, lng = get_location(address)

        else:
            lat = st.sidebar.number_input("Latitude:", value=43.064)
            lng = st.sidebar.number_input("Longitude:", value=-89.407)

    except Exception as e:
        st.error(f"No address >>> {e}")

    myzone = st.sidebar.radio("Ratio around the location (km)", [None, 2, 4], key='visibility')
    soc_prediction(lat, lng, myzone)

def layers_deprec(lat, lng):
    '''

    :param lat:
    :param lng:
    :return:
    '''
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


#map=map22(tmp_path)
#folium_static(map)
#upload1()