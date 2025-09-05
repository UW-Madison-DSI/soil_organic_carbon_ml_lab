import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

######################################################
######################################################
######################################################
# Cargar el dataset
df = pd.read_csv('final_conus_v2.csv')
features=['total_precipitation', 'min_temperature',
       'mean_temperature', 'max_temperature', 'dem', 'slope', 
       'aspect','hillshade', 'bd_mean', 
       'clay_mean', 'om_mean', 'ph_mean', 'sand_mean','silt_mean']
target='soil_organic_carbon'


soil_armonized_complete=df#[varsb]

# Convert columns to categorical data type
soil_armonized_complete['land_use'] = soil_armonized_complete['land_use'].astype('category')
soil_armonized_complete['land_cover'] = soil_armonized_complete['land_cover'].astype('category')

flag=True
try:
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

except Exception as e:
    print('Exception here ---',e)
    flag=False
##################################################
##################################################
##################################################
rom sklearn.model_selection import GridSearchCV

if falg is True:
    try:
        rf_model = RandomForestRegressor(n_estimators=500, max_features=10, random_state=42)
        rf_model.fit(X_cali, y_cali)

        # Predict on the calibration data
        rf_predict = rf_model.predict(X_cali)

        # Calculate goodness of fit
        gof_rf_predict = r2_score(y_cali, rf_predict)

        # Calibration
        print("Goodness of fit:", gof_rf_predict)


        # Define the grid of hyperparameters for Random Forest
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Define the grid of hyperparameters for XGBoost
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        # Perform grid search for Random Forest
        rf_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
        rf_grid_search.fit(X_cali, y_cali)

        # Perform grid search for XGBoost
        #xgb_grid_search = GridSearchCV(estimator=XGBRegressor(random_state=42), param_grid=xgb_param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
        #xgb_grid_search.fit(X_cali, y_cali)

        # Get the best models
        best_rf_model = rf_grid_search.best_estimator_
        #best_xgb_model = xgb_grid_search.best_estimator_


        # Predict on the test set
        rf_pred = best_rf_model.predict(X_cali)
        #xgb_pred = best_xgb_model.predict(X_cali)

        # Calculate R-squared value for both models
        rf_r2 = r2_score(y_cali, rf_pred)
        #xgb_r2 = r2_score(y_cali, xgb_pred)

        print(f'Random Forest R-squared: {rf_r2}')
        #print(f'XGBoost R-squared: {xgb_r2}')


        cali_predictions = rf_model.predict(X_cali)
        vali_predictions = rf_model.predict(X_vali)

        # Calculate R² score
        r2_cali = r2_score(y_cali, cali_predictions)
        r2_vali = r2_score(y_vali, vali_predictions)
        # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
        mse_cali = mean_squared_error(y_cali, cali_predictions)
        rmse_cali = mse_cali ** 0.5
        mse_vali = mean_squared_error(y_vali, vali_predictions)
        rmse_vali = mse_vali ** 0.5

        # Calculate Mean Absolute Error (MAE)
        mae_cali = mean_absolute_error(y_cali, cali_predictions)
        mae_vali = mean_absolute_error(y_vali, vali_predictions)

        print("========== RF Regression Performance ==========")
        # Print the performance metrics
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


        outp = {
            "Metric": ["R²", "MSE", "RMSE", "MAE"],
            "Calibration": [r2_cali, mse_cali, rmse_cali, mae_cali],
            "Validation": [r2_vali, mse_vali, rmse_vali, mae_vali]
        }

        # Create a DataFrame
        df_outp = pd.DataFrame(putp)
        print(df_outp)
    except Exception as e:
        print("Exception here ---2---", e)

# Print the DataFrame
print("=== RF Regression Performance WI ===")
print(df)


#from mlxtend.regressor import StackingRegressor
#from sklearn.linear_model import LinearRegression

# Create an ensemble model
#stacked_model = StackingRegressor(
#    regressors=[rf_model, xgb_model],
#    meta_regressor=LinearRegression()
#)

# Train the ensemble model
#stacked_model.fit(X_cali, y_cali)

# Predict on the test set using the ensemble model
#stacked_pred = stacked_model.predict(X_vali)

# Calculate R-squared value for the ensemble model
#stacked_r2 = r2_score(y_vali, stacked_pred)
#print(f'Stacked Model R-squared: {stacked_r2}')


##################################################
##################################################
##################################################
