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

from functions import *
from app_predictions import *
from app_sample_data import *
from map_layers import *
from modeling_from_user import *

def local_css(file_name: str):
    """Load a CSS file if it exists; fail gracefully otherwise."""
    try:
        # Try relative to script dir first, then CWD
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            return
        st.info(f"Note: CSS file not found: {file_name}")
    except Exception as e:
        st.warning(f"Could not load CSS ({file_name}): {e}")

def main():
    '''

    :return:
    '''
    import streamlit as st

    # put your CSS inside a triple-quoted string
    CUSTOM_CSS = """
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Red+Hat+Text:ital,wght@0,300..700;1,300..700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:ital,wght@0,300..900;1,300..900&family=Red+Hat+Text:ital,wght@0,300..700;1,300..700&display=swap');

    /* Font styles */
    :root {
        --font-family-sans-serif: "Red Hat Text", FreeSerif, serif;
        --RedHatDisplay: "Red Hat Display", sans-serif, bold;

        /* Color palette */
        --primary-color: #c5050c;
        --accent-red: #9B0000;
        --background-grey: #F7F7F7;
        --link-blue: #0479A8;
        --text-grey: #333333;
        --chunk-white: #fff;

        /* Element backgrounds and colors */
        --navbar-bg: var(--primary-color);
        --body-bg: var(--background-grey);
        --body-color: var(--text-grey);
        --link-color: var(--link-blue);
        --code-bg: var(--chunk-white);
        --code-color: var(--accent-red);
    }

    /* Applying background and text colors */
    body {
        background-color: var(--body-bg);
        color: var(--body-color);
    }

    /* Header styling */
    h1, h2 {
        color: var(--text-grey);
        font-family: var(--RedHatDisplay);
        font-weight: 800;
    }

    /* Navbar styles */
    .navbar-title {
        font-family: var(--RedHatDisplay);
        font-weight: 800;
    }

    /* Hyperlink styles */
    a {
        color: var(--link-color);
    }

    /* Styles for code elements */
    code {
        background-color: var(--code-bg);
        color: var(--code-color);
    }
    """

    # inject it once at the top of your app
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

    st.sidebar.title("Functionalities")

    sidebar_object = st.sidebar.radio('Please choose the option', ('My SOC models',
                                                                   'SOC sample data' #"SOC maps"
                                                                  ), key=100)

    if sidebar_object == 'SOC sample data':
        observed_soil_dynamic_properties()
    elif sidebar_object == 'My SOC models':
        modeling()
    else:
        map_layers_prediction()


if __name__ == "__main__":
    # Streamlit app layout
    main()
    st.write("---")

    # Add a note at the bottom of the page with links to the GitHub repo and survey
    st.markdown("""
    ### About this App

    This application is part of an ongoing cyber-infraestructure on soil organic carbon. We collected data from 
    public resources including GEE to predict soil organic carbon. Our goal is to make science accesible for 
    scientist that aim to impact in soil science by making tools easy to use to interact with and test hypothesis. You 
    can find the source code and contribute to the project via the GitHub repository. We also welcome your feedback 
    to improve the app. Please take a moment to fill out a short survey.

    - [GitHub Repository](https://github.com/UW-Madison-DSI/soil_organic_carbon_ml_lab.git)
    - [Survey Link](https://uwmadison.co1.qualtrics.com/jfe/form/SV_0PwdnBLdCjFPvlY)
    """)
