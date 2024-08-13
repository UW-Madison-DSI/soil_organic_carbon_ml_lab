import streamlit as st

from src.app_functions.functions import *
from src.app_functions.app_predictions import *
from src.app_functions.app_sample_data import *

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

    sidebar_object = st.sidebar.radio('Please choose the option', ("SOC Assistant",'SOC Observations'), key=100)

    if sidebar_object == 'SOC Assistant':
        map_layers_prediction()
    else:
        observed_soil_dynamic_properties()


if __name__ == "__main__":
    main()