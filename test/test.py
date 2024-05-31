import os
import pytest
from fastapi.testclient import TestClient
from main import app, PredictionRequest

# Create a TestClient for our FastAPI app
client = TestClient(app)

# Test data for a successful prediction
test_data = {
    "Depth": 1.0,
    "tmax": 30.0,
    "tmin": 15.0,
    "prcp": 10.0,
    "lc": 0.5,
    "clay": 20.0,
    "silt": 40.0,
    "sand": 40.0,
    "dem": 100.0,
    "slope": 5.0,
    "aspect": 180.0,
    "hillshade": 200.0,
    "twi": 8.0,
    "mrvbf": 2.0,
    "bulk_density": 1.3
}

def test_make_model_prediction():
    # Test a valid prediction request
    response = client.post('/v1/prediction', json=test_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "soil_organic_carbon" in json_response
    assert "soil_organic_carbon_stock" in json_response
    assert isinstance(json_response["soil_organic_carbon"], float)
    assert isinstance(json_response["soil_organic_carbon_stock"], float)

def test_prediction_request_validation():
    # Test a prediction request with missing fields
    incomplete_data = test_data.copy()
    del incomplete_data['Depth']
    response = client.post('/v1/prediction', json=incomplete_data)
    assert response.status_code == 422

def test_prediction_internal_server_error():
    # Test the prediction endpoint when an exception occurs (e.g., model file not found)
    response = client.post('/v1/prediction', json=test_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Model file not found"}

