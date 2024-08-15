<h1 align="center">

  Cyber infraestructure for Soil Organic Carbon Stock Forecasting Model
  <br>
</h1>

<h4 align="center">Random Forest Predictive Model Served as API build with FastAPI. This model predicts the soil organic carbon and soil organic carbon stocks in CONUS based on the soil dynamic properties of soil.
</h4>

Huang et al., 2019

## Key Features

This machine learning model predicts socs. So here are the key features of this project:
* Visualization of data https://socforecastingjhlab.streamlit.app
* The model is supported under a backend API built with `FastAPI` through the `POST` method, it asks the soil data as `JSON` format and returns its socs prediction in the same format. The API is served in posit connect: https://connect.doit.wisc.edu/soil_organic_carbon_prediction/v1/prediction see example below to test out.

* The `Dockerfile` saves all required information to run the model in another machines through a container. Just running the `initializer.sh` is enough to turn the whole system on.

* The `src` dir contains all the scripts required to update the model parameters.


<h1> Front-End Stack</h1>

Currently, the project is on development phase.  

## How To Use our API

To clone and run this application, follow these steps


``` EXAMPLE
import requests

url = 'https://connect.doit.wisc.edu/soil_organic_carbon_prediction/v1/prediction'

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

data = {
    "depth_cm": 5,
    "total_precipitation": 665.71,
    "mean_temperature": 19.702397,
    "dem": 1646.838135,
    "slope": 1.61754,
    "aspect": 25.694852,
    "hillshade": 178.0,
    "bd_mean": 1.317050,
    "clay_mean": 15.225413,
    "om_mean": 0.063926,
    "ph_mean": 7.000336,
    "sand_mean": 55.055664,
    "land_use": 5,
    "land_cover": 5
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("Prediction successful!")
    print(response.json())
else:
    print(f"Failed to get prediction: {response.status_code}")
    print(response.text)

```
## The API response should look as follows:
```
Prediction successful!
{'soil_organic_carbon': 1.5868605375289917, 'soil_organic_carbon_stock': 10.449873354762794}
```
## Resources

This software uses the following data and packages:

- [FastAPI](https://fastapi.tiangolo.com)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)
