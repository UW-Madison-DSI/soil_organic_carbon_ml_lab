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
* The model is supported under a backend API built with `FastAPI` through the `POST` method, it asks the soil data as `JSON` format and returns its socs prediction in the same format. The API is served in posit connect: https://connect.doit.wisc.edu/content/36e66cbf-6df9-42d8-9f68-8789f773ccf8 see example below to test out.

* The `Dockerfile` saves all required information to run the model in another machines through a container. Just running the `initializer.sh` is enough to turn the whole system on.

* The `src` dir contains all the scripts required to update the model parameters.


<h1> Front-End Stack</h1>

Currently, the project is on development phase.  

## How To Use

To clone and run this application, follow these steps

```bash
# Clone this repository
$ git clone https://github.com/UW-Madison-DSI/cyberinfraestructure_dsp_forecasting.git

# Go into the repository
$ cd cyberinfraestructure_dsp_forecasting

# Install requirements

$ pip install -r requirements.txt
$ pip install -r api/requirements.txt

# Install Backend dependencies

$ pip install fastapi


# Click on `POST` method

# Click on `Try it out`

# Replace the `Request Body` with a soil, it must have a json format, here is an example:
```

``` EXAMPLE
{
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



```
The API is a post
[API](https://connect.doit.wisc.edu/soil_organic_carbon_prediction/v1/prediction)

## Resources

This software uses the following data and packages:

- [FastAPI](https://fastapi.tiangolo.com)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)
