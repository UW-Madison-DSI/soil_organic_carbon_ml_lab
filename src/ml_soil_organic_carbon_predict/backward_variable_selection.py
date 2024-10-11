import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import json
import joblib
import pandas as pd
import time
import itertools

start_time = time.time()

df1 = pd.read_csv('final_conus_v2.csv')
features=[
      #'depth_cm',
      'total_precipitation',
      #'land_use', 'land_cover', 
      #'land_cover_class','land_use_class', 
      'min_temperature',
      #'max_temperature', 'mean_temperature', 
      #'dem', 
      'slope', 
      'aspect',
      'hillshade', 
      #'sand_mean',
      #'bd_mean', 
      'clay_mean',
      #'silt_mean',
      #'om_mean', 
      #'ph_mean',
      'sand_mean',
      'land_use',
      #'land_cover'
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
grouped = filtered_data.groupby(['latitude', 'longitude', 'depth_cm'])

# Select the first row of each group (arbitrarily choosing the first)
reduced_data = grouped.first().reset_index()

# Perform the train-test split with stratification
train_data, val_data = train_test_split(reduced_data, test_size=0.35, random_state=50, stratify=reduced_data['soil_id'])


# Create calibration and validation DataFrames
cali = train_data#[['Soil_ID', 'lc2', 'Land_Cover']]
vali = val_data#[['Soil_ID', 'lc2', 'Land_Cover']]

import itertools


features = [
        #'soil_organic_carbon',
        'depth_cm',
        'total_precipitation',
        'land_use', 'land_cover', 
        #'land_cover_class','land_use_class', 
        'min_temperature',
        'max_temperature', 
        'mean_temperature', 
        'dem', 
        'slope', 'aspect',
        'hillshade', 'bd_mean', 
        'clay_mean', 'om_mean', 'ph_mean', 'sand_mean',
        'silt_mean']

X_cali = cali#[features]
y_cali = cali[target]

X_vali = vali#[features]
y_vali = vali[target]

def evaluate_features(feature_subset):
    print("===================== Subset ", list(feature_subset), "=====================")
    X_cali_subset = cali[list(feature_subset)]
    X_vali_subset = vali[list(feature_subset)]

    # Use LinearRegression for faster training and evaluation
    rf_model = RandomForestRegressor(n_estimators=500, max_features=10, random_state=42)
    rf_model.fit(X_cali_subset, y_cali)

    cali_predictions = rf_model.predict(X_cali_subset)
    r2_cali = r2_score(y_cali, cali_predictions)
    


    vali_predictions = rf_model.predict(X_vali_subset)
    r2_vali = r2_score(y_vali, vali_predictions)
    print("R2 calubration ",r2_cali)
    print("R2 validation ",r2_vali)
    return r2_cali

def save_to_csv(data, filename='features.csv'):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Features', 'R-squared'])
        writer.writerows(data)

featsfinal = []
for r in range(5, 9):
    for combo in itertools.combinations(features, r):
        r2 = evaluate_features(combo)
        if 0.6 <= r2 <= 0.85:
            print(f"Found a valid feature combination: {combo} (R-squared: {r2:.4f})")
            featsfinal.append([combo, r2])

save_to_csv(featsfinal)