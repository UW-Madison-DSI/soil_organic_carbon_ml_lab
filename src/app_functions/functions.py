import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
from scipy.stats import t
import requests
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


import streamlit as st


maptilerkey = st.secrets['MAP_TILER_KEY']

def histogram_var(data, var):
    '''

    :param data:
    :param var:
    :return:
    '''
    var_ref = var.replace('_', ' ').replace('norm', '').replace('mean', '').replace('om', 'Organic matter').replace('bd', 'Bulk density').capitalize()
    fig = px.histogram(data, x=var, title=f"{var_ref} - Distribution")
    st.plotly_chart(fig)

def trend_time_series(dframe, depth_c, select_state):
    '''

    :param dframe:
    :param depth_c:
    :param select_state:
    :return:
    '''
    df = dframe
    if 'ALL' not in depth_c:
        df = dframe[dframe['depth_cm'] == int(depth_c.split('-')[-1])]

    try:
        if select_state != 'ALL':
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
            plt.title('Median Soil Organic Carbon across the years')
            st.pyplot(plt)

        df = df.groupby(by='year').agg({'soil_organic_carbon': 'median'})
        df = df.reset_index()

        df['DESAdd'] = ExponentialSmoothing(df['soil_organic_carbon'], trend='add').fit().fittedvalues.shift(-1)
        df['EWMA3'] = df['soil_organic_carbon'].ewm(alpha=2/3, adjust=False).mean()
        df['SES3'] = SimpleExpSmoothing(df['soil_organic_carbon']).fit(smoothing_level=.1,
                                                         optimized=False).fittedvalues.shift(-1)
        fig = go.Figure()

        # Add Sales line
        fig.add_trace(go.Scatter(x=df['year'], y=df['soil_organic_carbon'], mode='lines+markers', name='Obs'))
        # Add Trend line
        fig.add_trace(go.Scatter(x=df['year'], y=df['EWMA3'], mode='lines', name='EWMA3'))
        fig.add_trace(go.Scatter(x=df['year'], y=df['SES3'], mode='lines', name='SES3'))
        # Update layout for autoscaling x-axis
        fig.update_layout(
            title=f'Median Soil organic carbon trend {depth_c.split("-")[-1]} cm depth',
            xaxis_title='year',
            yaxis_title='Soil Organic Carbon',
            xaxis=dict(
                rangeslider=dict(visible=False),
                autorange=True
            )
        )
        # Display the plotly plot in Streamlit
        st.plotly_chart(fig)
    except Exception as e:
        st.error(e)
