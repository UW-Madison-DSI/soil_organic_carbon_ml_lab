## Overview
This script `predict_soc.py` trains and evaluates RandomForest and XGBoost models to predict soil organic carbon based on various environmental features. It performs data preprocessing, model training, and validation, and saves the models and predictions.

## Requirements
- Python 3.8
- Dependencies: numpy, pandas, scikit-learn, joblib, xgboost

## For developers

### 1. Clone the Repository
First, clone this repository to your local machine and locate at the script father folder.

```bash
git clone https://github.com/UW-Madison-DSI/soil_organic_carbon_ml_lab.git
cd src/ml_soil_organic_carbon_predict
run python predict_soc.py
```
## 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
# For macOS/Linux:
python3 -m venv env

# For Windows:
python -m venv env
```
```bash
# For macOS/Linux:
source env/bin/activate

# For Windows:
.\env\Scripts\activate
```
## 5. Install dependencies
You can install the project’s dependencies via requirements.txt
```bash
pip install -r requirements.txt
```

## 4. Run the code
```bash
run python predict_soc.py
```