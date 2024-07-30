#!/bin/bash

set -e

echo "Starting deployment script..."

pip install -U rsconnect-python

echo "Configuring Posit Connect server..."
rsconnect add -i --insecure --server https://connect.doit.wisc.edu --name sdp_CONUS --api-key $POSIT_API_KEY

echo "Deploying the Streamlit app..."
rsconnect deploy streamlit --insecure -n sdp_CONUS --entrypoint app.py .

echo "Deployment script completed successfully."