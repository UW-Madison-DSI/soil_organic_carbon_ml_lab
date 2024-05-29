#!/bin/bash

# Check if RSCONNECT_API_KEY is set
if [ -z "$RSCONNECT_API_KEY" ]; then
  echo "Error: RSCONNECT_API_KEY is not set."
  exit 1
fi

# Install rsconnect-python package
pip install -U rsconnect-python

export REQUESTS_CA_BUNDLE="$(pwd)/custom_bundle.crt"

# Deploy without specifying the custom certificate bundle
rsconnect deploy fastapi \
  --server https://connect.doit.wisc.edu/ \
  --api-key "$RSCONNECT_API_KEY" \
  ./api_socs_forecast/
