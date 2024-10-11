import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

import joblib
import pandas as pd
import time
import xgboost as xgb

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


"""
Instructions to run this script:
1. Ensure you have Python 3 installed along with the required libraries. You can install the dependencies using:
   pip install numpy pandas scikit-learn joblib xgboost and you are located at the script path
2. Place the dataset 'final_conus_v2.csv' in the same directory as this script or update the file path accordingly.
3. Run the script by executing the following command in your terminal:
   python <script_name>.py
4. The output, including the trained models and predictions, will be saved in the 'outputs_best_2/' directory.
"""

start_time = time.time()

df = pd.read_csv('final_conus_v2.csv')
features=['depth_cm','total_precipitation', #'min_temperature','max_temperature',
       'mean_temperature',  'dem', 'slope', 
       'aspect','hillshade', 'bd_mean', 'land_cover','land_use',
       'clay_mean', #'om_mean', 
       'ph_mean', 'sand_mean','silt_mean']
target='soil_organic_carbon'


soil_armonized_complete=df[df[target].between(0,100)]#[varsb]

# Convert columns to categorical data type
soil_armonized_complete['land_use'] = soil_armonized_complete['land_use'].astype('category')
soil_armonized_complete['land_cover'] = soil_armonized_complete['land_cover'].astype('category')

# Count instances in each class
class_counts = soil_armonized_complete['soil_id'].value_counts()

# Filter classes with at least two instances
filtered_data = soil_armonized_complete[soil_armonized_complete['soil_id'].isin(class_counts[class_counts > 1].index)]

# Perform the train-test split with stratification
train_data, val_data = train_test_split(filtered_data, test_size=0.25, random_state=50, stratify=filtered_data['soil_id'])


# Create calibration and validation DataFrames
cali = train_data#[['Soil_ID', 'lc2', 'Land_Cover']]
vali = val_data#[['Soil_ID', 'lc2', 'Land_Cover']]


# Extract features and target variable for calibration set
X_cali = cali[features]
y_cali = cali[target]

X_vali = vali[features]
y_vali = vali[target]


rf_model = RandomForestRegressor(n_estimators=500, max_features=10, random_state=42)
rf_model.fit(X_cali, y_cali)
joblib.dump(rf_model, 'outputs_best_2/rf_model_conus_v3.joblib')

print('Input >>',len(soil_armonized_complete))
# Define the parameter grid for XGBoost
param_grid_xgb = {
    'xgbregressor__n_estimators': [100, 150, 200],
    'xgbregressor__max_depth': [3, 6, 9],
    'xgbregressor__learning_rate': [0.01, 0.1, 0.2],
    'xgbregressor__subsample': [0.8, 1.0],
    'xgbregressor__colsample_bytree': [0.8, 1.0]
}

# Initialize the XGBoost Regressor
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Create a pipeline with standard scaler
pipeline_xgb = make_pipeline(StandardScaler(), xgb_regressor)

# Initialize GridSearchCV
grid_search_xgb = GridSearchCV(estimator=pipeline_xgb, param_grid=param_grid_xgb, cv=10, n_jobs=-1, verbose=2, scoring='r2')

# Fit the GridSearchCV
grid_search_xgb.fit(X_cali, y_cali)
print("--- %s seconds ---" % (time.time() - start_time))
# Get the best model
best_xgb_model = grid_search_xgb.best_estimator_

# Save the best XGBoost model
joblib.dump(best_xgb_model, 'outputs_best_2/xgb_model_optimized_conus_v2.joblib')

# Print feature importances if available
if hasattr(best_xgb_model.named_steps['xgbregressor'], 'feature_importances_'):
    importances = best_xgb_model.named_steps['xgbregressor'].feature_importances_
    feature_importance_df_xgb = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance_df_xgb)
else:
    print("The XGBoost model does not have feature importances.")


from sklearn.ensemble import VotingRegressor

# Initialize the RandomForestRegressor and XGBRegressor
rf_model = joblib.load('outputs_best_2/rf_model_conus_v3.joblib')
xgb_model = joblib.load('outputs_best_2/xgb_model_optimized_conus_v2.joblib')

# Combine both models using VotingRegressor
voting_regressor = VotingRegressor(estimators=[('rf', rf_model), ('xgb', xgb_model)])

# Fit the ensemble model
voting_regressor.fit(X_cali, y_cali)

# Save the ensemble model
joblib.dump(voting_regressor, 'outputs_best_2/voting_regressor_optimized_conus_v2.joblib')

# Predictions with the ensemble model
cali_predictions = voting_regressor.predict(X_cali)
vali_predictions = voting_regressor.predict(X_vali)

# Performance metrics
r2_cali = r2_score(y_cali, cali_predictions)
r2_vali = r2_score(y_vali, vali_predictions)
mse_cali = mean_squared_error(y_cali, cali_predictions)
rmse_cali = mse_cali ** 0.5
mse_vali = mean_squared_error(y_vali, vali_predictions)
rmse_vali = mse_vali ** 0.5
mae_cali = mean_absolute_error(y_cali, cali_predictions)
mae_vali = mean_absolute_error(y_vali, vali_predictions)

print("========== Voting Regressor Performance ==========")
print("Calibration:")
print("R²:", r2_cali)
print("MSE:", mse_cali)
print("RMSE:", rmse_cali)
print("MAE:", mae_cali)

print("Validation:")
print("R²:", r2_vali)
print("MSE:", mse_vali)
print("RMSE:", rmse_vali)
print("MAE:", mae_vali)

try:
    all_predictions = voting_regressor.predict(filtered_data[features])
    filtered_data['predict_ensembled']=all_predictions
    filtered_data.to_csv("outputs_best_2/predictions_optimized_rf_2.csv")
except Exception as e:
    print("Exception in saving the predictions ",e)

