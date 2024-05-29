from fastapi import FastAPI
import joblib
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame


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
    soil_organic_carbon_stock: float

def get_model() -> Pipeline:
    """
    Input: model path
    Output: model
    """
    loaded_rf = joblib.load("rf_model.joblib")
    return loaded_rf

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    """
    Input: data as JSON
    Output: Dataframe with the same info
    """
    dictionary = {key: [value] for key, value in class_model.dict().items()}
    df = DataFrame(dictionary)
    return df

def get_prediction(request: PredictionRequest) -> str:
    """
    Input: request. PredictionRequest instance used to compute the prediction of the model
    Output: prediction. Prediction of the model
    """
    try:
        data_to_predict = transform_to_dataframe(request)
        loaded_rf = get_model()
        predict = loaded_rf.predict(data_to_predict)
        return predict[0]
    except Exception as e:
        return -99.0

app = FastAPI(docs_url='/')

@app.post('/v1/prediction')
def make_model_prediction(request: PredictionRequest):
    """
    Input: Prediction request
    Output: Prediction response.
    """
    return PredictionResponse(soil_organic_carbon_stock=get_prediction(request))