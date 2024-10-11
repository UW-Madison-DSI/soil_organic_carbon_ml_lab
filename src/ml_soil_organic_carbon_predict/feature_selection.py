# Extract features and target variable for calibration set
#X_cali = cali[features]
#y_cali = cali[target]

#X_vali = vali[features]
#y_vali = vali[target]
#print(len(X_vali))

print("==========================================================")
print("Here in choosing the best model")
print("==========================================================")

import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt

def getBest(k):
    tic = time.time()
    
    results = []
    
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model


def processSubset(feature_set):
    # Fit the model on the given subset of features
    X = cali[list(feature_set)]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    MSE = mean_squared_error(y, predictions)
    r2_cali = r2_score(y, predictions)
    return {"model": model, "MSE": MSE,"R2": r2_cali, "features": feature_set}


def backward(predictors):
    tic = time.time()
    results = []
    
    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(combo))
    
    models = pd.DataFrame(results)

    
    toc = time.time()
    print('------------------------------')
    print(f"Processed {models.shape[0]} models on {len(predictors)-1} predictors in {toc-tic:.2f} seconds.")
    print('------------------------------')
    best_model = models.loc[models['R2'].between(.6,.8)]

    if best_model:
        return best_model
    else:
        pass

f = [
    'depth_cm', 'total_precipitation', 'land_use', 'land_cover',
    'min_temperature', 'max_temperature', 'mean_temperature', 
    'dem', 'slope', 'aspect', 'hillshade', 'clay_mean', 
    'ph_mean', 'sand_mean', 'silt_mean'
]

X = cali[f]
y = cali[target]

models_bwd = pd.DataFrame(columns=["MSE", "model"], index=range(1, len(X.columns)))

tic = time.time()
predictors = X.columns
results = []

t=5
while len(predictors)>1:
    print("Here ", t)
    best_model = backward(predictors)
    results.append(best_model)
    predictors = list(best_model["features"])
    models_bwd.loc[len(predictors)] = best_model
    t+=1

toc = time.time()
print(f"Total time taken: {toc-tic:.2f} seconds")

# Convert results to a DataFrame
final_models_df = pd.DataFrame(results)

# Save the results DataFrame to a CSV file
final_models_df.to_csv("results.csv", index=False)

# Select the best model with the minimum MSE
best_overall_model = final_models_df.loc[final_models_df['MSE'].idxmin()]

print("Best model selected:")
print(best_overall_model)

# Save the best model's details to a JSON file
best_model_details = {
    "features": best_overall_model['features'],
    "MSE": best_overall_model['MSE']
}

with open("best_model.json", "w") as json_file:
    json.dump(best_model_details, json_file)

# The summary of the best model (Random Forest does not have a summary method like OLS)
print("Features:")
for feature in best_overall_model['features']:
    print(f" - {feature}")

# Print the MSE of the best overall model
print(f"MSE: {best_overall_model['MSE']}")