#!/bin/bash

set -e

echo "Starting deployment script..."

# Install rsconnect-python
echo "Installing rsconnect-python..."
pip install -U rsconnect-python

# Deploy the Streamlit app
echo "Deploying the Streamlit app..."
rsconnect deploy streamlit --server https://connect.doit.wisc.edu --api-key $POSIT_API_KEY ./

echo "Deployment script completed successfully."