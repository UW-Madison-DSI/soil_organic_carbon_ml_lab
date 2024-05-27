from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO
from joblib import load
import logging
from sklearn.pipeline import Pipeline
import joblib

logging.basicConfig(level=logging.DEBUG)

class PredictionRequest(BaseModel):
    Depth: float
    tmax: float
    tmin: float
    prcp: float
    lc: float
    clay: float
    silt: float
    sand: float
    dem: float
    slope: float
    aspect: float
    hillshade: float
    twi: float
    mrvbf: float

class PredictionResponse(BaseModel):
    socs: float

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    """
    Input: Patient data as JSON
    Output: Dataframe with the same info
    """
    features = ['Depth', 'tmax', 'tmin', 'prcp', 'lc', 'clay', 'silt', 'sand', 'dem', 'slope', 'aspect', 'hillshade', 'twi', 'mrvbf']
    dictionary = {key: [value] for key, value in class_model.dict().items()}
    df = DataFrame(dictionary)
    df = df[features]
    return df

def get_model() -> Pipeline:
    """
    Input: model path
    Output: model
    """
    rf_model = load('rf_model.joblib')
    return rf_model

def get_prediction(request: PredictionRequest) -> float:
    """
    Input: request. PredictionRequest instance used to compute the prediction of the model
    Output: prediction. Prediction of the model
    """
    try:
        data_to_predict = transform_to_dataframe(request)

        model = get_model()

        prediction = model.predict(data_to_predict)[0]
        return prediction
        
    except Exception as e:
        return str(e)


app = FastAPI(docs_url='/')

@app.post('/v1/prediction')
def make_model_prediction(request: PredictionRequest):
    """
    Input: Prediction request
    Output: Prediction response. The prediction is 0 for Benign samples and 1 for Malignant samples, so we return the name instead the number
    """
    return PredictionResponse(socs=get_prediction(request))
