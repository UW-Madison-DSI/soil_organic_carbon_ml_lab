import streamlit as st
import requests
import pandas as pd
import folium
from folium.plugins import HeatMap
from pygwalker.api.streamlit import StreamlitRenderer


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

from src.app_functions.functions import *
from src.app_functions.app_predictions import *
from src.app_functions.app_sample_data import *
from src.app_functions.map_layers import *
from src.app_functions.modeling_from_user import *

def local_css(file_name):
    '''

    :param file_name:
    :return:
    '''
    with open(file_name) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


def main():
    '''

    :return:
    '''
    local_css("frontend/css/streamlit.css")
    st.sidebar.title("Functionalities")

    sidebar_object = st.sidebar.radio('Please choose the option', ('My SOC models',
                                                                   'SOC sample data',
                                                                   "SOC maps"), key=100)

    if sidebar_object == 'SOC sample data':
        map_layers_prediction()
    elif sidebar_object == 'My SOC models':
        modeling()
    else:
        observed_soil_dynamic_properties()


if __name__ == "__main__":
    # Streamlit app layout
    main()
    st.write("---")

    # Add a note at the bottom of the page with links to the GitHub repo and survey
    st.markdown("""
    ### About this App

    This application is part of an ongoing cyberinfraestructure on soil organic carbon. You can find the source code and contribute to the project via the GitHub repository. We also welcome your feedback to improve the app. Please take a moment to fill out a short survey.

    - [GitHub Repository](https://github.com/UW-Madison-DSI/soil_organic_carbon_ml_lab.git)
    - [Survey Link](https://uwmadison.co1.qualtrics.com/jfe/form/SV_0PwdnBLdCjFPvlY)
    """)
