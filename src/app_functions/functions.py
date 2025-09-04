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


def scatterplot_var(data, x_var, y_var):
    """
    Generate and display a scatter plot for the specified variables.
    Args:
        data (DataFrame): The dataset containing the variables.
        x_var (str): The variable to plot on the x-axis.
        y_var (str): The variable to plot on the y-axis.
    """
    # Check if both variables are numeric
    if not pd.api.types.is_numeric_dtype(data[x_var]):
        st.write(f"{x_var} is not a numeric variable.")
        return

    if not pd.api.types.is_numeric_dtype(data[y_var]):
        st.write(f"{y_var} is not a numeric variable.")
        return

    # Remove missing values
    data_clean = data[[x_var, y_var]].dropna()

    if data_clean.empty:
        st.write(f"No valid data available for plotting {x_var} and {y_var}.")
        return

    # Rename variables for display
    x_var_ref = (x_var.replace('_', ' ')
                 .replace('norm', '')
                 .replace('mean', '')
                 .replace('om', 'Organic matter')
                 .replace('bd', 'Bulk density')
                 .capitalize())

    y_var_ref = (y_var.replace('_', ' ')
                 .replace('norm', '')
                 .replace('mean', '')
                 .replace('om', 'Organic matter')
                 .replace('bd', 'Bulk density')
                 .capitalize())

    # Generate scatter plot
    fig = px.scatter(data_clean, x=x_var, y=y_var,
                     title=f"{x_var_ref} vs {y_var_ref} %")
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


# Assuming 'filtered_data' contains 'year', 'soil_organic_carbon', 'temperature', and 'precipitation'
import statsmodels.api as sm


from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def add_time_series_prediction_with_predictors(
    data: pd.DataFrame,
    predictor_variables: list,
    n_periods: int = 5,
    agg: str = "mean"  # how to combine multiple rows per year
) -> None:
    """
    Forecast soil_organic_carbon with exogenous predictors, allowing multiple rows per year.
    Uses a simple step index for the model (0..T-1) and keeps 'year' only for plotting.
    """
    variable_name = "soil_organic_carbon"

    # Basic checks
    if 'year' not in data.columns or variable_name not in data.columns:
        st.warning("Data must contain 'year' and 'soil_organic_carbon' columns.")
        return
    if not all(col in data.columns for col in predictor_variables):
        st.error(f"Missing one or more predictor variables: {predictor_variables}")
        return

    # Ensure numeric years and aggregate duplicates
    df = data.copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Choose aggregation for multiple rows per year
    if agg == "median":
        ts_data = (df.groupby('year')[[variable_name] + predictor_variables]
                     .median()
                     .sort_index()
                     .dropna())
    else:
        ts_data = (df.groupby('year')[[variable_name] + predictor_variables]
                     .mean()
                     .sort_index()
                     .dropna())

    if ts_data.empty or len(ts_data) < 3:
        st.warning("Not enough aggregated yearly data (need ≥ 3 years) after cleaning.")
        return

    # Real years for display
    years_obs = ts_data.index.to_numpy().astype(int)
    last_year = int(years_obs[-1])
    future_years = np.arange(last_year + 1, last_year + 1 + n_periods, dtype=int)

    # Endog/exog for the model — use a simple 0..T-1 step index internally
    y = ts_data[variable_name].to_numpy()
    X = ts_data[predictor_variables].to_numpy()

    # Fit SARIMAX (ARIMAX-like)
    try:
        model = SARIMAX(
            endog=y,
            exog=X,
            order=(1, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        # Future exogenous: repeat the last observed row (naive)
        future_exog = np.tile(X[-1, :], (n_periods, 1))

        # Forecast n steps ahead
        # Forecast n steps ahead
        fcst = results.get_forecast(steps=n_periods, exog=future_exog)
        mean_forecast = fcst.predicted_mean#.to_numpy()  # shape (n_periods,)
        conf_int = fcst.conf_int()#.to_numpy()  # (n_periods x 2)

        # -------- Plot --------
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Observations
        ax.plot(years_obs, y, label='Observations', marker='o')

        # Forecast
        ax.plot(future_years, mean_forecast, label='Predictions', linestyle='--', marker='x')

        # CI band
        ax.fill_between(
            future_years,
            conf_int[:, 0],  # lower bound
            conf_int[:, 1],  # upper bound
            alpha=0.2,
            color="orange",
            label="95% CI"
        )

        ax.set_title(f"Time Series Prediction for {variable_name} with Predictors")
        ax.set_xlabel("Year")
        ax.set_ylabel(variable_name)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        st.pyplot(fig)

        # Forecast table with explicit years + CI
        out = pd.DataFrame({
            'year': future_years,
            'predicted_mean': mean_forecast,
            'lower_ci': conf_int[:, 0],
            'upper_ci': conf_int[:, 1]
        })
        st.markdown(f"**Predicted {variable_name} for the next {n_periods} years:**")
        st.dataframe(out)


    except Exception as e:
        st.error(f"An error occurred during time series analysis: {e}")
