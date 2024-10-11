import sys
import time
import json
import pandas as pd
import ee
import geemap

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='ee-moros2')

"""
Script to download terrain variables from GEE. 
Assumes you have an account in GEE to authenticate.
"""

# Load the input CSV file
soil_data = pd.read_csv('grid_wi_input.csv')

# Extract unique soil points
unique_soil_points = soil_data[['soil_id', 'longitude', 'latitude']].drop_duplicates()

# Define the DEM and derived terrain products
dem = ee.Image('USGS/3DEP/10m').select('elevation')
slope = ee.Terrain.slope(dem)
aspect = ee.Terrain.aspect(dem)
hillshade = ee.Terrain.hillshade(dem)


# Function to extract mean values from Earth Engine Image
def extract_values(image, point, scale=30):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=scale,
        maxPixels=1e9
    )
    return stats.getInfo()


# Function to get DEM and terrain parameters for a location
def get_dem_params(longitude, latitude):
    point = ee.Geometry.Point([longitude, latitude])
    dem_values = extract_values(dem, point)
    slope_values = extract_values(slope, point)
    aspect_values = extract_values(aspect, point)
    hillshade_values = extract_values(hillshade, point)

    return {
        'dem': dem_values.get('elevation'),
        'slope_deg': slope_values.get('slope'),
        'aspect_deg': aspect_values.get('aspect'),
        'hillshade': hillshade_values.get('hillshade')
    }


# Batch processing and saving results
start_time = time.time()
batch_size = 2000
batch = []
file_index = 0
path = "../dem_batch_"

for i, row in unique_soil_points.iterrows():
    params = get_dem_params(row['longitude'], row['latitude'])

    new_record = {
        row['soil_id']: {
            'longitude': row['longitude'],
            'latitude': row['latitude'],
            **params
        }
    }

    batch.append(new_record)

    if len(batch) >= batch_size:
        with open(f'{path}_{file_index}.jsonl', 'w') as outfile:
            outfile.write("\n".join(json.dumps(record) for record in batch))
        file_index += 1
        batch = []

if batch:
    with open(f'{path}_{file_index}.jsonl', 'w') as outfile:
        outfile.write("\n".join(json.dumps(record) for record in batch))

print(f"--- {time.time() - start_time} seconds ---")