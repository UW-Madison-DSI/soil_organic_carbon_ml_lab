if [ -z "$RSCONNECT_API_KEY" ]; then
  echo "Error: RSCONNECT_API_KEY is not set."
  exit 1
fi

pip install -U rsconnect-python

rsconnect deploy streamlit -n sdp_CONUS --entrypoint app.py ./
