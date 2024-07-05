from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandas import DataFrame
import joblib
from sklearn.pipeline import Pipeline
from typing import List, Any


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
    bulk_density: float


class PredictionResponse(BaseModel):
    soil_organic_carbon: float
    soil_organic_carbon_stock: float


def transform_to_dataframe(requests: List[PredictionRequest]) -> DataFrame:
    """
    Transforms a list of PredictionRequest instances into a DataFrame.

    Args:
    requests (List[PredictionRequest]): List of prediction requests

    Returns:
    DataFrame: DataFrame containing the input data for the model
    """
    dictionary = {key: [getattr(req, key) for req in requests] for key in requests[0].dict().keys()}
    df = DataFrame(dictionary)
    return df


def get_prediction(request: PredictionRequest) -> List[Any]:
    """
    Generates a prediction using a pre-trained model.

    Args:
    request (PredictionRequest): Prediction request data

    Returns:
    List[Any]: List containing the soil organic carbon and soil organic carbon stock
    """
    try:
        features = ['Depth', 'tmax', 'tmin', 'prcp', 'lc', 'clay', 'silt', 'sand', 'dem', 'slope', 'aspect', 'hillshade', 'twi', 'mrvbf']
        data_to_predict = transform_to_dataframe([request])
        loaded_rf = joblib.load("rf_model.joblib")
        prediction = loaded_rf.predict(data_to_predict[features])
        soil_organic_carbon = prediction[0]
        soil_organic_carbon_stock = soil_organic_carbon * data_to_predict.iloc[0]['Depth'] * data_to_predict.iloc[0]['bulk_density']
        return [soil_organic_carbon, soil_organic_carbon_stock]
    except Exception as e:
        return [-1,-1]


app = FastAPI(docs_url='/')

@app.post('/v1/prediction', response_model=PredictionResponse)
def make_model_prediction(request: PredictionRequest):
    """
    Endpoint to make predictions on soc and then soc stocks

    Args:
    the data is a manual input now, in the future the sdp will be estimated based on location and date

    Returns:
    PredictionResponse: Soil organic carbon estimate (based on RF from JH) and soc stock: soc*BD*depth
    """
    prediction = get_prediction(request)
    return PredictionResponse(soil_organic_carbon=prediction[0], soil_organic_carbon_stock=prediction[1])
