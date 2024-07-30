#!/bin/bash

set -e

# Install rsconnect-python
pip install -U rsconnect-python

# Add Posit Connect server
rsconnect add -i --server https://connect.doit.wisc.edu --name sdp_forecasting --api-key $POSIT_API_KEY

# Deploy the Streamlit app
rsconnect deploy streamlit -n wisc --entrypoint app.py .
