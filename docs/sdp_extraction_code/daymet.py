import sys
import time
import json
import pandas as pd
import daymetpy
import os
import re
import dask
from dask import delayed, compute


"""
Script to download temporal data from daymetpy, for the SOC prediction purposes 
the data is aggregated annually in terms of the mean for maximum and minimum temperature and the sum for the total precipitation 
"""

# Append the necessary path
sys.path.append(r"..")

# Load the input CSV file
soil_armonized_neon = pd.read_csv('grid_wi_input.csv')

# Extract unique soil points
soil_unique_points1 = soil_armonized_neon[['soil_id', 'longitude', 'latitude']].drop_duplicates()

# Set the path to save batches
year = 1990
path = f"batch_{year}/temporal_vars"

# Initialize variables
batch_size = 1000
batch = []
total_extractions = 0
start_time = time.time()

# Define the function to process a single soil point
@delayed
def process_soil_point(val, year):
    try:
        params = daymetpy.daymet_timeseries(lon=val['longitude'], lat=val['latitude'],
                                            start_year=year, end_year=year)
        # Prepare data to save
        new = {
            val['soil_id']: {
                'longitude': val['longitude'],
                'latitude': val['latitude'],
                'year': year,
                'min_temperature': params['tmin'].mean(),
                'mean_temperature': params['tmin'].mean() + params['tmax'].mean(),
                'max_temperature': params['tmax'].mean(),
                'prcp': params['prcp'].sum()
            }
        }
        return new
    except Exception as e:
        print(f"Error processing soil_id {val['soil_id']}: {e}")
        return None



try:
    print(f"Processing year: {year}")
    
    # List to store delayed tasks
    tasks = []
    
    for i, (k, val) in enumerate(soil_unique_points1.iterrows()):
        # Add delayed tasks
        task = process_soil_point(val, year)
        tasks.append(task)
        total_extractions += 1

        # Process tasks in batches
        if len(tasks) >= batch_size:
            results = compute(*tasks)  # Execute the delayed tasks
            batch = [res for res in results if res is not None]

            output_file = f'{path}/dynamicprop_1990_{file_index}.json'
            with open(output_file, 'w') as outfile:
                json.dump(batch, outfile, indent=4)
            print(f"Saved batch {file_index} with {len(batch)} entries to {output_file}")

            # Clear batch and increment file index
            batch = []
            tasks = []
            file_index += 1

    # Save any remaining data in the final batch
    if tasks:
        results = compute(*tasks)
        batch = [res for res in results if res is not None]
        
        output_file = f'{path}/dynamicprop_batch_{file_index}.json'
        with open(output_file, 'w') as outfile:
            json.dump(batch, outfile, indent=4)
        print(f"Saved final batch {file_index} with {len(batch)} entries to {output_file}")

    # Print the total time taken for the year
    elapsed_time = time.time() - start_time
    print(f"Year {year} completed in --- {elapsed_time:.2f} seconds ---")
    print(f"Total extractions so far: {total_extractions}")

except Exception as e:
    print(e)

# Print the total time taken overall
total_elapsed_time = time.time() - start_time
print(f"--- Total time: {total_elapsed_time:.2f} seconds ---")
print(f"Total extractions: {total_extractions}")
