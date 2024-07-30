#!/bin/bash

set -e

echo "Starting deployment script..."

# Install rsconnect-python
echo "Installing rsconnect-python..."
pip install -U rsconnect-python

echo "Configuring Posit Connect server..."
rsconnect add --server https://connect.doit.wisc.edu --name sdp_CONUS --api-key $POSIT_API_KEY

# Deploy the Streamlit app with the specified name and entry point
echo "Deploying the Streamlit app..."
rsconnect deploy streamlit --name sdp_CONUS --entrypoint app.py .

echo "Deployment script completed successfully."