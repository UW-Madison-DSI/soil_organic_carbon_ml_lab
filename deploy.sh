if [ -z "$POSIT_API_KEY" ]; then
  echo "Error: $POSIT_API_KEY is not set."
  exit 1
fi

pip install -U rsconnect-python

rsconnect add -i --server https://connect.doit.wisc.edu --name sdp_CONUS --api-key $POSIT_API_KEY
rsconnect deploy streamlit -n sdp_CONUS --entrypoint app.py ./
