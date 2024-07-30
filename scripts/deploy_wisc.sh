pip install -U rsconnect-python
rsconnect add -i --server https://connect.doit.wisc.edu --name sdp_forecasting --api-key $POSIT_API_KEY
rsconnect deploy streamlit -n wisc --entrypoint app.py .