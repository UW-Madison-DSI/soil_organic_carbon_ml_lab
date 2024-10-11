# Soil Organic Carbon ML Lab

This repository contains educational content and tools for performing **machine learning** (ML) modeling on **soil organic carbon** data. The goal is to provide an interactive learning experience for users to explore and apply various ML techniques in environmental data modeling, specifically focused on soil health metrics.

Visit our **Cyber-Infraestructure** [here](https://soilorganiccarbon-ml-lab.streamlit.app/)

Link to our public API of soil organic carbon model for CONUS [here](https://connect.doit.wisc.edu/soil_organic_carbon_prediction/)


## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

The **Soil Organic Carbon ML Lab** is designed to help users learn how to:
- Analyze soil organic carbon data using various machine learning models.
- Experiment with data preprocessing, model training, and evaluation in an interactive environment.

The lab is equipped with various tools and educational resources that allow users to build predictive models and explore geospatial data related to soil properties.

## Installation

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Recommended: A virtual environment tool such as `venv` or `conda`.

### Clone the Repository

To get started, clone this repository to your local machine:

```
git clone https://github.com/UW-Madison-DSI/soil_organic_carbon_ml_lab.git
cd soil_organic_carbon_ml_lab
```

### Virtual Environment

Set up a virtual environment (recommended):

```
# For venv
python3 -m venv venv
source venv/bin/activate  # On Windows, use 'venv\\Scripts\\activate'

# For conda
conda create --name soil_ml python=3.8
conda activate soil_ml
```

### Install Dependencies

Install the required Python packages:

```
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebooks

This repository includes several Jupyter notebooks that guide users through different stages of machine learning modeling for soil organic carbon data. To run the notebooks:


Then, open the notebook of your choice and follow the instructions within.

### Running the Streamlit App

If the project includes a Streamlit-based web application, run the app with:

```
streamlit run app.py
```

This will launch a web interface for interacting with the soil organic carbon ML models.

### Example Notebooks

- **Data Preprocessing.ipynb**: Steps to clean, prepare, and explore the soil organic carbon dataset.
- **ML Modeling.ipynb**: Build and evaluate machine learning models such as Random Forest and XGBoost to predict soil carbon levels.
- **Visualization.ipynb**: Visualize trends in soil organic carbon across different geographical regions.

## Data

The soil organic carbon data used in this project can be sourced from [public datasets](https://example.com) or can be loaded from your local directory. 

- **states_shape/States_shapefile.shp**: Shapefile used for geospatial analysis of states.
- **data/sample_soc_observations/**: Contains sample data for soil organic carbon observations. This data was retrived in a big effor of data extraction from public sources.

Ensure that the data is structured correctly in the required format before running the models.

## Features

- 
## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (\`git checkout -b feature-branch\`).
3. Commit your changes (\`git commit -m "Add some feature"\`).
4. Push to the branch (\`git push origin feature-branch\`).
5. Open a pull request.

### Reporting Issues

If you encounter any bugs or have questions, please open an issue in the repository.

## License

This project is licensed under the MIT License.

## Contact

For questions or inquiries, please contact:

- **Jingyi Huang** - Assistant Professor, Soil Sciences Dept at UW-Madison
  Email: [jhuang426@wisc.edu](mailto:jhuang426@wisc.edu)
- **Maria Oros** - Data Scientist, UW-Madison DSI
  Email: [moros2@wisc.edu](mailto:moros2@wisc.edu)
