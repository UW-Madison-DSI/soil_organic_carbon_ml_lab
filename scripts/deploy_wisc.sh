#!/bin/bash

set -e

echo "Starting deployment script..."

# Install rsconnect-python
echo "Installing rsconnect-python..."
pip install -U rsconnect-python

# Deploy the Streamlit app
echo "Deploying the Streamlit app..."
rsconnect deploy streamlit --server https://connect.doit.wisc.edu --name sdp_CONUS --api-key $POSIT_API_KEY --entrypoint app.py . --no-verify-ssl

echo "Deployment script completed successfully."