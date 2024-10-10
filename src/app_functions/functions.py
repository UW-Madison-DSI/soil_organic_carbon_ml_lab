import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import streamlit as st
import requests


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

    try:
        if select_state != 'ALL':
            df_aggregated = df.groupby(['State_Name', 'year']).agg(
                mean_soil_organic_carbon=('soil_organic_carbon', 'median'),
                sem_soil_organic_carbon=('soil_organic_carbon', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
                count=('soil_organic_carbon', 'count')
            ).reset_index()

            df_pivot = df_aggregated.pivot(index='year', columns='State_Name', values='mean_soil_organic_carbon')

            # Fit SARIMA model for each state
            models = {select_state: sm.tsa.SARIMAX(df_pivot[select_state], order=(1, 1, 1),
                                                   seasonal_order=(1, 1, 1, 12)).fit()}

            # Compute confidence intervals
            df_aggregated['t_critical'] = df_aggregated['count'].apply(lambda x: t.ppf(0.975, x - 1))
            df_aggregated['ci_lower'] = df_aggregated['mean_soil_organic_carbon'] - df_aggregated['t_critical'] * \
                                        df_aggregated['sem_soil_organic_carbon']
            df_aggregated['ci_upper'] = df_aggregated['mean_soil_organic_carbon'] + df_aggregated['t_critical'] * \
                                        df_aggregated['sem_soil_organic_carbon']

            predictions = {}
            conf_ints = {}
            for state, model in models.items():
                pred = model.get_forecast(steps=2)
                predictions[state] = pred.predicted_mean
                conf_ints[state] = pred.conf_int()

            # Plot the results with confidence intervals
            plt.figure(figsize=(14, 7))
            for state in df_pivot.columns:
                plt.plot(df_pivot.index, df_pivot[state], label=f'{state} - Actual')
                plt.plot(predictions[state].index, predictions[state], label=f'{state} - Forecast')
                plt.fill_between(conf_ints[state].index,
                                 conf_ints[state].iloc[:, 0],
                                 conf_ints[state].iloc[:, 1], alpha=0.2, label=f'{state} - CI')

            for state in df_aggregated['State_Name'].unique():
                state_data = df_aggregated[df_aggregated['State_Name'] == state]
                plt.fill_between(state_data['year'], state_data['ci_lower'], state_data['ci_upper'], alpha=0.1,
                                 label=f'{state} - Historical CI')

            plt.legend()
            plt.xlabel('Year')
            plt.ylabel('Soil Organic Carbon')
            plt.title('Median Soil Organic Carbon across the years')
            st.pyplot(plt)

        # General trend analysis using Exponential Smoothing
        df = df.groupby(by='year').agg({'soil_organic_carbon': 'median'}).reset_index()
        df['DESAdd'] = ExponentialSmoothing(df['soil_organic_carbon'], trend='add').fit().fittedvalues.shift(-1)
        df['EWMA3'] = df['soil_organic_carbon'].ewm(alpha=2 / 3, adjust=False).mean()
        df['SES3'] = SimpleExpSmoothing(df['soil_organic_carbon']).fit(smoothing_level=.1,
                                                                       optimized=False).fittedvalues.shift(-1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['year'], y=df['soil_organic_carbon'], mode='lines+markers', name='Obs'))
        fig.add_trace(go.Scatter(x=df['year'], y=df['EWMA3'], mode='lines', name='EWMA3'))
        fig.add_trace(go.Scatter(x=df['year'], y=df['SES3'], mode='lines', name='SES3'))

        fig.update_layout(
            title=f'Median Soil Organic Carbon Trend {depth_c.split("-")[-1]} cm Depth',
            xaxis_title='Year',
            yaxis_title='Soil Organic Carbon',
            xaxis=dict(rangeslider=dict(visible=False), autorange=True)
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(e)

def init():
    '''

    Returns:

    '''
    turkey_coord = [39.653098, -99.101648]
    turkey_map_normal = folium.Map(location=turkey_coord, zoom_start=5.5)
    df = pd.read_parquet('data/sample_soc_observations/final_conus_v2.parquet', engine='pyarrow')
    HeatMap(data=df[['latitude', 'longitude', 'soil_organic_carbon']], radius=5).add_to(turkey_map_normal)

    heat_data = df[['latitude', 'longitude', 'soil_organic_carbon']].values.tolist()
    HeatMap(data=heat_data, radius=5).add_to(turkey_map_normal)

    norm = matplotlib.colors.Normalize(vmin=df['soil_organic_carbon'].min(), vmax=df['soil_organic_carbon'].max())
    cmap = matplotlib.cm.get_cmap('YlOrRd')

    for index, row in df.iterrows():
        color = matplotlib.colors.rgb2hex(cmap(norm(row['soil_organic_carbon'])))

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['soil_organic_carbon'] / 10,
            color='blue',
            fill=True,
            fill_color=color,
            fill_opacity=0.6
        ).add_to(turkey_map_normal)

    st_folium(turkey_map_normal, width=700, height=500)


def init22():

    # Streamlit app title
    st.title("Map with Rectangles around Points")

    # Define your point (latitude, longitude)
    lat, lon = 40.7128, -74.0060  # Example: New York City

    # Create a small bounding box around the point
    delta = 0.01  # Defines the size of the rectangle
    bounding_box = [[lat - delta, lon - delta], [lat + delta, lon + delta]]

    # Create a folium map centered on the point
    m = folium.Map(location=[lat, lon], zoom_start=12)

    # Add the rectangle to the map
    folium.Rectangle(bounds=bounding_box, color='blue', fill=True, fill_opacity=0.2).add_to(m)

    # Add a marker at the center point
    folium.Marker([lat, lon], popup="New York City").add_to(m)

    # Display the map in Streamlit using streamlit-folium
    st_data = st_folium(m, width=700, height=500)
    #st_folium(st_data, width=700, height=500)


def pygwalkertool():
    # Adjust the width of the Streamlit page
    st.set_page_config(
        page_title="Use Pygwalker In Streamlit",
        layout="wide"
    )

    # Add Title
    st.title("Use Pygwalker In Streamlit")

    from datetime import datetime
    # You should cache your pygwalker renderer, if you don't want your memory to explode
    @st.cache_resource
    def get_pyg_renderer() -> "StreamlitRenderer":
        df = pd.read_parquet('data/sample_soc_observations/final_conus_v2.parquet', engine='pyarrow')

        df['date']=df['year'].apply(lambda x: f'{int(x)}-01-01', '%Y-%m-%d')
        st.dataframe(df)
        # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
        return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")

    renderer = get_pyg_renderer()
    renderer.explorer()