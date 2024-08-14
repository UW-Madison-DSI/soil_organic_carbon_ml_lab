#!/bin/bash

set -e

echo "Starting deployment script..."

pip install -U rsconnect-python

echo "Deployment script completed successfully."

rsconnect deploy fastapi --server https://connect.doit.wisc.edu/ --api-key $POSIT_API_KEY ./api
