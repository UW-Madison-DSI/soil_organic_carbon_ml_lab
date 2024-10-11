import streamlit as st
import pandas as pd
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Custom DecisionTree and RandomForest classes
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self.get_best_split(X, y, num_features)
            if best_split['score'] is not None:
                left_tree = self.fit(best_split['left_X'], best_split['left_y'], depth + 1)
                right_tree = self.fit(best_split['right_X'], best_split['right_y'], depth + 1)
                return {'feature': best_split['feature'], 'threshold': best_split['threshold'],
                        'left_tree': left_tree, 'right_tree': right_tree}
        return np.mean(y)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['feature']] < tree['threshold']:
                return self._predict_single(x, tree['left_tree'])
            else:
                return self._predict_single(x, tree['right_tree'])
        else:
            return tree

    def get_best_split(self, X, y, num_features):
        best_split = {'score': None}
        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] < threshold
                right_idx = ~left_idx
                left_y, right_y = y[left_idx], y[right_idx]
                score = self.calculate_mse(left_y, right_y)
                if best_split['score'] is None or score < best_split['score']:
                    best_split.update({
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_X': X[left_idx], 'right_X': X[right_idx],
                        'left_y': left_y, 'right_y': right_y,
                        'score': score
                    })
        return best_split

    def calculate_mse(self, left_y, right_y):
        mse = lambda y: np.mean((y - np.mean(y)) ** 2)
        total_size = len(left_y) + len(right_y)
        return (len(left_y) / total_size) * mse(left_y) + (len(right_y) / total_size) * mse(right_y)


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.tree = tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)  # averaging for regression


# Function to perform linear regression based on formula
def linear_regression_from_formula(formula: str, data: pd.DataFrame):
    try:
        # Parse the formula and split into dependent (outcome) and independent (x1, x2...) variables
        y, X = patsy.dmatrices(formula, data)

        # Fit the modeling_soc_conus using Ordinary Least Squares (OLS)
        model = sm.OLS(y, X)
        result = model.fit()

        # Return the fitted modeling_soc_conus
        return result
    except patsy.PatsyError as e:
        raise ValueError(f"Error in formula: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error in regression: {str(e)}")


def pre_processing_data_uploaded(data):
    # Validate missing values
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        st.warning(f"The dataset contains {missing_values} missing values.")
        return data

    # Validate if 'soil_organic_carbon' column exists
    if 'soil_organic_carbon' not in data.columns:
        st.error("The 'soil_organic_carbon' column is missing from the dataset.")
        return data

    # Validate that 'soil_organic_carbon' values are between 0 and 100
    if not data['soil_organic_carbon'].between(0, 100).all():
        st.error("Some values in the 'soil_organic_carbon' column are outside the range [0, 100].")
        return data

    # If the dataset is valid, show a success message
    st.success("Dataset is valid!")

    # Display a histogram of 'soil_organic_carbon'
    fig, ax = plt.subplots()
    data['soil_organic_carbon'].hist(ax=ax, bins=10, color='blue', edgecolor='black')
    ax.set_title('Soil Organic Carbon Distribution')
    ax.set_xlabel('Soil Organic Carbon')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    return data


def modeling():
    # Streamlit app layout
    st.title('My own soil organic carbon models')

    sidebar_object = st.radio('Please choose the data option you would like to use',
                                      ('Use the default dataset of soil dynamic properties',
                                       'Upload my soil organic carbon data'), key=100100)
    flag = True
    if sidebar_object == 'Upload my soil organic carbon data':
        st.write("Please be sure the dataset is with a column called `soil_organic_carbon` , and other numerical variables thaat can play the role of predictors.")
        # File uploader for CSV
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            # Read the CSV file
            data0 = pd.read_csv(uploaded_file)
            data = pre_processing_data_uploaded(data0)
            st.write("### Data Preview")
            st.write(data.head())
        else:
            flag=False
            pass

    else:
        flag = True
        # Sample data if no file is uploaded
        data = pd.read_parquet('data/sample_soc_observations/final_conus_v2.parquet', engine='pyarrow')
        st.write("### Sample Data")
        st.write("The sample data is the one described in sample data section.")
        st.write(data[[x for x in data.columns if 'id' not in x and 'Unn' not in x and 'label' not in x]].head())

    if flag == True:
        # Display column names
        st.write("### Available Columns")
        st.write(", ".join(data[[x for x in data.columns if 'id' not in x and 'Unn' not in x and 'label' not in x]].columns))

        formula = st.text_input('Enter the formula for the modeling_soc_conus (e.g., soil_organic_carbon ~ aspect + bd_mean). The left '
                                'side is the independent or outcome variable:',
                                value='soil_organic_carbon ~ aspect + bd_mean')

        if '~' in formula:
            # Extract the part after the '~' to see if any predictors are specified
            response, predictors = formula.split('~')
            predictors = predictors.strip()

            # If no predictors are specified, use 'aspect' as the default
            if not predictors:
                st.warning("No predictors specified, using 'aspect' as the default predictor.")
                predictors = 'aspect'
                formula = response.strip() + ' ~ ' + predictors
                predictor_columns = ['aspect']
            else:
                if '+' in formula:
                    # Split the predictors part into individual columns and remove spaces
                    predictor_columns = [col.strip() for col in predictors.split('+')]
                if '*' in formula:
                    # Split the predictors part into individual columns and remove spaces
                    predictor_columns = [col.strip() for col in predictors.split('*')]
        else:
            st.error(
                "Invalid formula format. Please ensure the formula contains a '~' separating the response and predictors.")

        # Subset the data to include only the relevant columns
        try:
            # Identify all required columns for the formula
            required_columns = [response.strip()] + predictor_columns
            required_columns = list(set(required_columns))

            subset_data = data[required_columns]

        except KeyError as e:
            st.error(f"Missing columns in the dataset: {e}")

        if st.checkbox('Show Correlation Heatmap of The Variables'):
            st.write("### Correlation Heatmap")
            numeric_cols = subset_data.select_dtypes(include=[np.number])  # Select only numerical columns
            if len(numeric_cols.columns) > 1:
                corr_matrix = numeric_cols.corr()  # Calculate the correlation matrix

                fig, ax = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
                ax.set_title('Correlation Heatmap')
                st.pyplot(fig)
            else:
                st.warning("Not enough numerical variables to show correlation heatmap.")
                pass


        y, X = patsy.dmatrices(formula, subset_data)
        y = y.ravel()

        model_type = st.radio('Please choose the modeling you want to perform',
                              ('Linear Regression',
                               'Random Forest' #not showed yet
                               ), key=100101)
        if model_type == 'Random Forests':
            st.write("It might take few minutes.")
            if st.button('Run Model'):
                model = RandomForest(n_estimators=10, max_depth=3)
                model.fit(np.array(X), y)

                # Make predictions
                predictions = model.predict(np.array(X))

                # Display the predictions
                st.write('### Predictions')
                st.write(predictions)

                # Plot actual vs predicted values
                fig, ax = plt.subplots()
                ax.scatter(y, predictions, color='blue')
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Actual vs Predicted (Random Forest)')
                st.pyplot(fig)
        else:
            if st.button('Run Model'):
                # Ask user to input the modeling_soc_conus formula
                # Perform regression when button is clicked
                try:
                    result = linear_regression_from_formula(formula, data)
                    st.write('### Linear Regression Results')
                    # R-squared
                    r_squared = result.rsquared
                    st.write(
                        f"The R-squared of the modeling_soc_conus is {r_squared:.4f}, indicating that the modeling_soc_conus explains {r_squared * 100:.2f}% of the variance in `soil_organic_carbon`.")

                    st.text(result.summary())

                    # Make predictions
                    predictions = result.predict(X)

                    # Plot actual vs predicted values
                    fig, ax = plt.subplots()
                    ax.scatter(data['soil_organic_carbon'], predictions, color='blue')
                    ax.plot([data['soil_organic_carbon'].min(), data['soil_organic_carbon'].max()],
                            [data['soil_organic_carbon'].min(), data['soil_organic_carbon'].max()],
                                'r--')
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Actual vs Predicted')
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")