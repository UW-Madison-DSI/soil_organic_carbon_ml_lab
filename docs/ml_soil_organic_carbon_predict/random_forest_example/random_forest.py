import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV,cross_val_score, KFold

import joblib
import pandas as pd
import time

start_time = time.time()

df1 = pd.read_csv('final_conus_v2.csv')
features=[     
        'total_precipitation',
        'land_cover',
        #'land_use', 
        'mean_temperature',
        'dem', 'hillshade',
        # 'aspect',
        #'clay_mean',
        'silt_mean'
        ]

target='soil_organic_carbon'

grouped = df1.groupby(['latitude', 'longitude', 'depth_cm'])

# Select the first row of each group (arbitrarily choosing the first)
df = grouped.first().reset_index()
soil_armonized_complete=df[df['soil_organic_carbon'].between(0,100)].copy()#[varsb]
soil_armonized_complete['depth_cm']=soil_armonized_complete['depth_cm'].astype(int)
# Convert columns to categorical data type
soil_armonized_complete['land_use'] = soil_armonized_complete['land_use'].astype('category')
soil_armonized_complete['land_cover'] = soil_armonized_complete['land_cover'].astype('category')

print("Input here is ", len(soil_armonized_complete))
# Count instances in each class
class_counts = soil_armonized_complete['soil_id'].value_counts()

# Filter classes with at least two instances
filtered_data = soil_armonized_complete[soil_armonized_complete['soil_id'].isin(class_counts[class_counts > 1].index)]

# Perform the train-test split with stratification
train_data, val_data = train_test_split(filtered_data, test_size=0.35, random_state=50, stratify=filtered_data['soil_id'])


# Create calibration and validation DataFrames
cali = train_data#[['Soil_ID', 'lc2', 'Land_Cover']]
vali = val_data#[['Soil_ID', 'lc2', 'Land_Cover']]


# Extract features and target variable for calibration set
X_cali = cali[features]
y_cali = cali[target]

X_vali = vali[features]
y_vali = vali[target]
print(len(X_vali))


print("==========================================================")
print("============ Training Random Forest Regressor ============")
print("==========================================================")

rf_model = RandomForestRegressor(n_estimators=500, max_features=10, random_state=42)
rf_model.fit(X_cali, y_cali)
joblib.dump(rf_model, 'outputs/rf_model_conus_v2.joblib')
print("--- %s seconds training---" % (time.time() - start_time))


# Set up cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and calculate MSE for each fold
mse_scores = -cross_val_score(rf_model, X_cali, y_cali, cv=cv, scoring='neg_mean_squared_error')

# Calculate mean and standard deviation of MSE scores
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f'Mean MSE: {mean_mse:.4f}')
print(f'Standard Deviation of MSE: {std_mse:.4f}')

# Check if the model is a RandomForest model
if hasattr(rf_model, 'feature_importances_'):
    # Get feature importances
    importances = rf_model.feature_importances_

    # If you have feature names, you can create a DataFrame for better readability
    # Replace 'feature_names' with your actual feature names

    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Print feature importances
    print(feature_importance_df)
else:
    print("The model does not have feature importances.")
#rf_model = RandomForestRegressor(n_estimators=500, max_features=10, random_state=42)
#rf_model.fit(X_cali, y_cali)
# Calibration

print("--- %s seconds ---" % (time.time() - start_time))

try:
	all_predictions = rf_model.predict(filtered_data[features])
	filtered_data['predict']=all_predictions
	filtered_data.to_csv("outputs/predictions_rf.csv")
except Exception as e:
	print("Exception in saving the predictions ",e)

#xgb_r2 = r2_score(y_cali, xgb_pred)

###########here in metrics of performance
try:
	cali_predictions = rf_model.predict(X_cali)
	vali_predictions = rf_model.predict(X_vali)
	#gof_rf_predict = r2_score(y_cali, cali_predictions)

    # Calibration
    #print("Goodness of fit:", gof_rf_predict)

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
except Exception as e:
	print("Exception here --->>> ", e)