import os

from fastapi.testclient import TestClient
from api_socs_forecast.main import app

client = TestClient(app)


connect_api_key = os.environ.get("RSCONNECT_API_KEY")
headers = {
    "Authorization": f"Key {connect_api_key}",
    "accept": "application/json",
    "Content-Type": "application/json"
  }

def test_null_predictions():
    response = client.post('/v1/prediction', headers=headers,
                                            json={"Depth": 0,
                                                  "tmax": 0,
                                                  "tmin": 0,
                                                  "prcp": 0,
                                                  "lc": 0,
                                                  "clay": 0,
                                                  "silt": 0,
                                                  "sand": 0,
                                                  "dem": 0,
                                                  "slope": 0,
                                                  "aspect": 0,
                                                  "hillshade": 0,
                                                  "twi": 0,
                                                  "mrvbf": 0,
                                                  "bulk_density": 0})

    assert response.status_code == 200
    assert type(response.json()['diagnostic']) is str

def test_random_prediction():
    response = client.post('/v1/prediction', headers=headers,
                                            json={"Depth": 5,
                                                 "tmax": 10.702739716,
                                                 "tmin": 0.5561643839,
                                                 "prcp": 753.0,
                                                 "lc": 9.0,
                                                 "clay": 10.0,
                                                 "silt": 35.0,
                                                 "sand": 55.0,
                                                 "dem": 189,
                                                 "slope": 5.69661e-05,
                                                 "aspect": 6.283185482,
                                                 "hillshade": 0.7853578925,
                                                 "twi": 11.223488808,
                                                 "mrvbf": 2.5688176155,
                                                "bulk_density": 1.88})
    assert response.status_code == 200
    assert type(response.json()['diagnostic']) is str
