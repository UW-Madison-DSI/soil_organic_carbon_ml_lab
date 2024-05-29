if [ -z "$RSCONNECT_API_KEY" ]; then
  echo "Error: RSCONNECT_API_KEY is not set."
  exit 1
fi

pip install -U rsconnect-python
# Deploy using the environment variable for the API key
rsconnect deploy fastapi --server https://connect.doit.wisc.edu/ --api-key $RSCONNECT_API_KEY ./api_socs_forecast/