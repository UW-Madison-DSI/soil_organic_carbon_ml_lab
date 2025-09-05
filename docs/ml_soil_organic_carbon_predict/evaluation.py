import numpy as np
import pandas as pd
import time
import joblib
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

"""
This is the code to evaluate multiple pre-trained models.

Instructions to run this script:
1. Ensure you have Python 3 installed along with the required libraries. You can install the dependencies using:
   pip install numpy pandas scikit-learn joblib xgboost
2. Place the dataset 'final_conus_v2.csv' in the same directory as this script or update the file path accordingly.
3. Run the script by executing the following command in your terminal:
   python <script_name>.py
4. The output, including the trained models and predictions, will be saved in the 'outputs_best_2/' directory.
"""

# Start timer
start_time = time.time()

# Load dataset
df = pd.read_csv('final_conus_v2.csv')

# Define features and target
features = [
    'depth_cm', 'total_precipitation', 'mean_temperature', 'dem', 
    'slope', 'aspect', 'hillshade', 'bd_mean', 'land_cover', 
    'land_use', 'clay_mean', 'ph_mean', 'sand_mean', 'silt_mean'
]
target = 'soil_organic_carbon'

# Filter and preprocess dataset
soil_data = df[df[target].between(0, 100)]
soil_data['land_use'] = soil_data['land_use'].astype('category')
soil_data['land_cover'] = soil_data['land_cover'].astype('category')

# Filter classes with at least two instances
class_counts = soil_data['soil_id'].value_counts()
filtered_data = soil_data[soil_data['soil_id'].isin(class_counts[class_counts > 1].index)]

# Perform train-test split with stratification
train_data, val_data = train_test_split(filtered_data, test_size=0.25, random_state=50, stratify=filtered_data['soil_id'])

# Extract features and target variables
X_train, y_train = train_data[features], train_data[target]
X_val, y_val = val_data[features], val_data[target]

# Load models
print("---------->>> Loading Models <<<----------")
try:
	rf_model = joblib.load('pretrained_models/rf_model_conus_v3.joblib')
	print("ok reading rf!")
except Exception as e:
	pass

try:
	voting_regressor = joblib.load('pretrained_models/voting_regressor_optimized_conus_v2.joblib')
	print("ok voting rf!")
except Exception as e:
	pass
try:
	xgb_model = joblib.load('pretrained_models/xgb_model_optimized_conus_v2.joblib')
	print("ok reading rf and xgboost!")
except Exception as e:
	pass



def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    print(f"========== {model_name} ==========")
    try:
	    # Predictions
	    train_predictions = model.predict(X_train)
	    val_predictions = model.predict(X_val)
	    
	    # Performance metrics
	    metrics = {
	        'R²': (r2_score, 'train_predictions', 'val_predictions'),
	        'MSE': (mean_squared_error, 'train_predictions', 'val_predictions'),
	        'RMSE': (lambda y_true, y_pred: mean_squared_error(y_true, y_pred) ** 0.5, 'train_predictions', 'val_predictions'),
	        'MAE': (mean_absolute_error, 'train_predictions', 'val_predictions')
	    }
	    
	    for metric_name, (metric_func, train_pred, val_pred) in metrics.items():
	        train_metric = metric_func(y_train, train_predictions)
	        val_metric = metric_func(y_val, val_predictions)
	        print(f"{metric_name} Calibration: {train_metric:.4f}")
	        print(f"{metric_name} Validation: {val_metric:.4f}")

	    print("ok getting the metrics the first one!")
    
    except Exception as e:
    	pass

# Evaluate models
evaluate_model(rf_model, X_train, y_train, X_val, y_val, "RF Model")
#evaluate_model(voting_regressor, X_train, y_train, X_val, y_val, "Voting Regressor")
evaluate_model(xgb_model, X_train, y_train, X_val, y_val, "XGBoost")

# Print runtime
print(f"Total runtime: {time.time() - start_time:.2f} seconds")
