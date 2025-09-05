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
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Optional, Literal


def add_time_series_prediction_with_predictors(
        data: pd.DataFrame,
        predictor_variables: List[str],
        n_periods: int = 5,
        agg: Literal["mean", "median"] = "mean",
        target_variable: str = "soil_organic_carbon",
        min_years_required: int = 3,
        model_order: tuple = (1, 0, 1),
        confidence_level: float = 0.95
) -> Optional[pd.DataFrame]:
    """
    Forecast a target variable using SARIMAX with exogenous predictors.

    Handles multiple rows per year by aggregating them, then fits a time series model
    to predict future values with confidence intervals.

    Args:
        data: DataFrame containing time series data
        predictor_variables: List of column names to use as predictors
        n_periods: Number of future periods to forecast
        agg: Aggregation method for multiple rows per year ("mean" or "median")
        target_variable: Name of the target variable column
        min_years_required: Minimum number of years needed for modeling
        model_order: ARIMA order tuple (p, d, q)
        confidence_level: Confidence level for prediction intervals

    Returns:
        DataFrame with predictions or None if modeling fails

    Raises:
        None (errors are displayed via Streamlit)
    """

    # Input validation
    if not _validate_inputs(data, predictor_variables, target_variable):
        return None

    # Prepare and aggregate data
    processed_data = _prepare_time_series_data(
        data, target_variable, predictor_variables, agg
    )

    if processed_data is None or len(processed_data) < min_years_required:
        st.warning(
            f"Insufficient data after cleaning. Need ≥ {min_years_required} years, got {len(processed_data) if processed_data is not None else 0}.")
        return None

    # Extract time components
    years_observed = processed_data.index.to_numpy().astype(int)
    future_years = _generate_future_years(years_observed, n_periods)

    # Prepare model inputs
    y = processed_data[target_variable].to_numpy()
    X = processed_data[predictor_variables].to_numpy()

    # Fit model and generate predictions
    try:
        predictions_df = _fit_and_predict(
            y, X, future_years, n_periods, model_order, confidence_level
        )

        # Create visualizations
        _create_forecast_plot(
            years_observed, y, future_years, predictions_df,
            target_variable, confidence_level
        )

        # Display results
        _display_results(predictions_df, target_variable, n_periods)

        return predictions_df

    except Exception as e:
        st.error(f"Model fitting failed: {str(e)}")
        return None


def _validate_inputs(
        data: pd.DataFrame,
        predictor_variables: List[str],
        target_variable: str
) -> bool:
    """Validate input data and parameters."""
    required_columns = ['year', target_variable]
    missing_required = [col for col in required_columns if col not in data.columns]

    if missing_required:
        st.error(f"Missing required columns: {missing_required}")
        return False

    missing_predictors = [col for col in predictor_variables if col not in data.columns]
    if missing_predictors:
        st.error(f"Missing predictor variables: {missing_predictors}")
        return False

    return True


def _prepare_time_series_data(
        data: pd.DataFrame,
        target_variable: str,
        predictor_variables: List[str],
        agg: str
) -> Optional[pd.DataFrame]:
    """Clean and aggregate time series data by year."""
    df = data.copy()

    # Convert year to numeric and clean
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Aggregate by year
    columns_to_aggregate = [target_variable] + predictor_variables
    aggregation_func = getattr(df.groupby('year')[columns_to_aggregate], agg)

    ts_data = (aggregation_func()
               .sort_index()
               .dropna())

    return ts_data if not ts_data.empty else None


def _generate_future_years(years_observed: np.ndarray, n_periods: int) -> np.ndarray:
    """Generate future year values for forecasting."""
    last_year = int(years_observed[-1])
    return np.arange(last_year + 1, last_year + 1 + n_periods, dtype=int)


def _fit_and_predict(
        y: np.ndarray,
        X: np.ndarray,
        future_years: np.ndarray,
        n_periods: int,
        model_order: tuple,
        confidence_level: float
) -> pd.DataFrame:
    """Fit SARIMAX model and generate predictions."""
    # Fit model
    model = SARIMAX(
        endog=y,
        exog=X,
        order=model_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Prepare future exogenous variables (naive approach: repeat last values)
    future_exog = np.tile(X[-1, :], (n_periods, 1))

    # Generate forecast
    forecast = results.get_forecast(steps=n_periods, exog=future_exog)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=1 - confidence_level)

    # Create results DataFrame
    return pd.DataFrame({
        'year': future_years,
        'predicted_mean': mean_forecast,
        'lower_ci': conf_int[:, 0],  # Use numpy indexing instead of iloc
        'upper_ci': conf_int[:, 1]
    })


def _create_forecast_plot(
        years_observed: np.ndarray,
        y_observed: np.ndarray,
        future_years: np.ndarray,
        predictions_df: pd.DataFrame,
        target_variable: str,
        confidence_level: float
) -> None:
    """Create and display forecast visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data
    ax.plot(years_observed, y_observed,
            label='Historical Data', marker='o', linewidth=2, markersize=6)

    # Plot predictions
    ax.plot(future_years, predictions_df['predicted_mean'],
            label='Forecast', linestyle='--', marker='x',
            linewidth=2, markersize=8, color='red')

    # Add confidence interval
    ci_percentage = int(confidence_level * 100)
    ax.fill_between(
        future_years,
        predictions_df['lower_ci'],
        predictions_df['upper_ci'],
        alpha=0.25,
        color="orange",
        label=f"{ci_percentage}% Confidence Interval"
    )

    # Styling
    ax.set_title(f"Time Series Forecast: {target_variable.replace('_', ' ').title()}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(target_variable.replace('_', ' ').title(), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    st.pyplot(fig)


def _display_results(
        predictions_df: pd.DataFrame,
        target_variable: str,
        n_periods: int
) -> None:
    """Display forecast results in a formatted table."""
    st.markdown(f"### 📊 Forecast Results")
    st.markdown(f"**Predicted {target_variable.replace('_', ' ').title()} for the next {n_periods} years:**")

    # Format the dataframe for better display
    display_df = predictions_df.copy()
    numeric_columns = ['predicted_mean', 'lower_ci', 'upper_ci']
    for col in numeric_columns:
        display_df[col] = display_df[col].round(3)

    # Rename columns for better presentation
    display_df.columns = ['Year', 'Predicted Value', 'Lower CI (95%)', 'Upper CI (95%)']

    st.dataframe(display_df, use_container_width=True)