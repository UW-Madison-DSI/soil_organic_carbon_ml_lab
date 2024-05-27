<h1 align="center">

  Cyber infraestructure for Soil Organic Carbon Stock Forecasting Model
  <br>
</h1>

<h4 align="center">Random Forest Predictive Model Served as API build with FastAPI. This model predicts the soil organic carbon stock based on properties of soil.
</h4>

Huang et al., 2019

## Key Features

This machine learning model predicts socs. So here are the key features of this project:

* The model is supported under a backend API built with `FastAPI` through the `POST` method, it asks the patients data as `JSON` format and returns its socs prediction in the same format. The API is served in posit connect: https://connect.doit.wisc.edu/content/36e66cbf-6df9-42d8-9f68-8789f773ccf8 see example below to test out.

* The `Dockerfile` saves all required information to run the model in another machines through a container. Just running the `initializer.sh` is enough to turn the whole system on.

* The `src` dir contains all the scripts required to update the model parameters.


<h1> Front-End Stack</h1>

Currently, the project is on **Front-End** phase. It is planned to be developed using the framework `Angular CLI`, which helps us to consume the REST API. The source code can be viewed in the directory `/static`. Here's how it looks


## How To Use

To clone and run this application, follow these steps

```bash
# Clone this repository
$ git clone https://github.com/UW-Madison-DSI/cyberinfraestructure_dsp_forecasting.git

# Go into the repository
$ cd cyberinfraestructure_dsp_forecasting

# Install requirements

$ pip install -r requirements.txt
$ pip install -r requirements_test.txt
$ pip install -r api/requirements.txt

# Install Backend dependencies

$ pip install uvicorn
$ pip install fastapi

# Run the server

$ uvicorn api.main:app

# Server is set to be constant, so run in your browser:

http://127.0.0.1:8001 

# Click on `POST` method

# Click on `Try it out`

# Replace the `Request Body` with a patient data, it must have a json format, here is an example:

{"Depth": 5,
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
"mrvbf": 2.5688176155}

# Click on execute and view (Or download) the results

```

## Resources

This software uses the following data and packages:

- [FastAPI](https://fastapi.tiangolo.com)
- [Docker](https://www.docker.com)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)
