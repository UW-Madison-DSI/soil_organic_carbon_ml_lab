import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import patsy
import statsmodels.api as sm


def polynomial_features_1d(x1d, degree=3):
    """x1d: shape (n,) -> returns Vandermonde with bias."""
    x1d = np.asarray(x1d).ravel()
    cols = [np.ones_like(x1d)]
    for d in range(1, degree + 1):
        cols.append(x1d ** d)
    return np.column_stack(cols)


def _safe_corrcoef(df_numeric):
    # numpy corrcoef expects rows as variables if rowvar=True; use rowvar=False
    return np.corrcoef(df_numeric.values, rowvar=False), list(df_numeric.columns)


def gradient_descent_regression(X, y, lr=0.01, epochs=1000, standardize=True):
    """
    NumPy-only GD for linear regression with:
      - optional feature standardization
      - bias term auto-added
      - history of loss and gradient norm
    Returns:
      beta_gd        : (p+1,) parameters [intercept, ...]
      history        : dict with 'loss', 'grad_norm' (lists)
      X_for_plotting : design matrix actually used for GD preds (with leading 1s)
      unstandardize  : function to map beta back to original feature scale (if needed)
    """
    X = np.asarray(X); y = np.asarray(y).ravel()

    # standardize features (not the target) for stable steps
    if standardize:
        mu = X.mean(axis=0); sigma = X.std(axis=0) + 1e-12
        Xs = (X - mu) / sigma
        def unstandardize(beta_std):
            # beta_std = [b0, b1, ..., bp] on standardized X
            b0 = beta_std[0] - np.sum(beta_std[1:] * (mu / sigma))
            b = beta_std[1:] / sigma
            return np.concatenate([[b0], b])
    else:
        Xs = X
        def unstandardize(beta_std):
            return beta_std

    # add bias
    Xb = np.c_[np.ones(Xs.shape[0]), Xs]
    n, p1 = Xb.shape
    beta = np.zeros(p1)

    history = {"loss": [], "grad_norm": []}

    for _ in range(epochs):
        preds = Xb @ beta
        err = preds - y
        loss = (err @ err) / n
        grad = (2.0 / n) * (Xb.T @ err)

        history["loss"].append(float(loss))
        history["grad_norm"].append(float(np.linalg.norm(grad)))

        beta -= lr * grad

    # matrix used for plotting predictions
    X_for_plotting = Xb
    # return coefficients in ORIGINAL scale
    beta_unstd = unstandardize(beta)

    return beta_unstd, history, X_for_plotting



# Function to perform linear regression based on formula
def linear_regression_from_formula(formula: str, data: pd.DataFrame):
    try:
        # Parse the formula and split into dependent (outcome) and independent (x1, x2...) variables
        y, X = patsy.dmatrices(formula, data)

        # Fit the model using Ordinary Least Squares (OLS)
        model = sm.OLS(y, X)
        result = model.fit()

        # Return the fitted model
        return result
    except patsy.PatsyError as e:
        raise ValueError(f"Error in formula: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error in regression: {str(e)}")


# Function to perform polynomial regression using statsmodels
def polynomial_regression_from_data(x_data, y_data, degree=3):
    try:
        # Create polynomial features
        X_poly = polynomial_features_1d(x_data, degree=degree)

        # Fit the model using OLS
        model = sm.OLS(y_data, X_poly)
        result = model.fit()

        return result, X_poly
    except Exception as e:
        raise ValueError(f"Error in polynomial regression: {str(e)}")


# Function to perform gradient descent regression using statsmodels for comparison
def gradient_descent_with_sm_comparison(X, y, lr=0.01, epochs=1000):
    """
    Returns:
      beta_gd         : GD params in ORIGINAL feature scale (incl. intercept)
      result_sm       : statsmodels OLS result (with intercept)
      X_for_plotting  : design matrix used for GD (with leading 1s, on standardized X)
    """
    X = np.asarray(X); y = np.asarray(y).ravel()

    # ensure OLS has intercept
    X_sm = np.c_[np.ones(X.shape[0]), X]

    # --- GD (with history) ---
    beta_gd, history, X_for_plotting = gradient_descent_regression(
        X, y, lr=lr, epochs=epochs, standardize=True
    )

    # --- OLS comparison ---
    result_sm = sm.OLS(y, X_sm).fit()

    return (beta_gd, result_sm, X_for_plotting, history)


from collections import Counter


def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        idx = np.argsort(dists)[:k]
        neighbors = y_train[idx]
        if np.issubdtype(y_train.dtype, np.integer):
            # classification
            preds.append(Counter(neighbors).most_common(1)[0][0])
        else:
            # regression
            preds.append(np.mean(neighbors))
    return np.array(preds)


def modeling():
    st.title('My own soil organic carbon models')

    sidebar_object = st.radio(
        'Please choose the data option you would like to use',
        ('Use the default dataset of soil dynamic properties',
         'Upload my soil organic carbon data'),
        key=100100
    )

    data = None
    if sidebar_object == 'Upload my soil organic carbon data':
        st.write(
            "Please be sure the dataset has a column called `soil_organic_carbon` and other numerical predictor columns.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data0 = pd.read_csv(uploaded_file)
            # If you have a preprocessor, apply it; else pass-through
            if 'pre_processing_data_uploaded' in globals():
                data = pre_processing_data_uploaded(data0)
            else:
                data = data0.copy()
            st.write("### Data Preview")
            st.write(data.head())
        else:
            st.info("Upload a CSV to continue.")
            return
    else:
        # Sample data path
        data = pd.read_parquet('data/sample_soc_observations/final_conus_v2.parquet', engine='pyarrow')
        st.write("### Sample Data")
        st.write("The sample data is the one described in sample data section.")
        st.write(data[[c for c in data.columns if 'id' not in c and 'Unn' not in c and 'label' not in c]].head())

    # Show available columns
    st.write("### Available Columns")
    usable_cols = [c for c in data.columns if ('id' not in c and 'Unn' not in c and 'label' not in c)]
    st.write(", ".join(usable_cols))

    # Modeling formula
    formula = st.text_input(
        'Enter the formula for the model (e.g., soil_organic_carbon ~ aspect + bd_mean). '
        'Left side is the dependent/outcome variable:',
        value='soil_organic_carbon ~ aspect + bd_mean'
    )

    # Parse formula to extract response and predictors
    if '~' not in formula:
        st.error("Invalid formula format. Use `y ~ x1 + x2`.")
        return

    response, rhs = formula.split('~', 1)
    response = response.strip()
    rhs = rhs.strip()

    if rhs == '':
        st.warning("No predictors specified; defaulting to 'aspect'.")
        rhs = 'aspect'
        formula = f"{response} ~ {rhs}"

    # Determine predictor columns (simple split on +; keep * if user wants interactions—patsy handles it)
    # For subset we'll include tokens that look like column names (letters, numbers, underscores)
    import re
    tokens = re.findall(r"[A-Za-z_]\w*", rhs)
    predictor_columns = [t for t in tokens if t in data.columns]

    # Subset data for speed/safety
    required_columns = sorted(set([response] + predictor_columns))
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        st.error(f"Missing columns in the dataset: {missing}")
        return

    subset_data = data[required_columns].dropna()

    # Optional correlation heatmap (Matplotlib only)
    if st.checkbox('Show Correlation Heatmap of the Variables'):
        st.write("### Correlation Heatmap")
        numeric_cols = subset_data.select_dtypes(include=[np.number])
        if numeric_cols.shape[1] > 1:
            corr, labels = _safe_corrcoef(numeric_cols)
            fig, ax = plt.subplots()
            im = ax.imshow(corr, vmin=-1, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
            ax.set_title('Correlation Heatmap')
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical variables to show correlation heatmap.")

    # Build design matrices with patsy (supports interactions in rhs)
    try:
        y_mat, X_mat = patsy.dmatrices(formula, subset_data, return_type='dataframe')
        y_vec = np.asarray(y_mat).ravel()
        X_np = np.asarray(X_mat)  # includes intercept column if specified by patsy
    except Exception as e:
        st.error(f"Error building design matrices from formula: {e}")
        return

    # Model choice
    model_options = ['Linear Regression (Statsmodels OLS)',
                     'Polynomial Regression (Statsmodels OLS)',
                     'Gradient Descent vs OLS Comparison']

    model_type = st.radio(
        'Please choose the modeling you want to perform',
        tuple(model_options),
        key=100101
    )

    # Additional options for polynomial regression
    if model_type == 'Polynomial Regression (Statsmodels OLS)':
        degree = st.slider('Polynomial Degree', min_value=2, max_value=6, value=3)

    run = st.button('Run Model')
    # Get feature names aligned with GD coefficients (Intercept + features)
    if hasattr(X_mat, 'columns'):
        # Does X_mat already include an intercept column?
        x_has_intercept = np.allclose(np.asarray(X_mat)[:, 0], 1.0)
        if x_has_intercept:
            feature_names = list(X_mat.columns)  # already includes Intercept
        else:
            feature_names = ['Intercept'] + list(X_mat.columns)
    else:
        # Fallback generic names
        feature_names = ['Intercept'] + [f'Feature_{i + 1}' for i in range(X_for_plotting.shape[1] - 1)]


    if model_type == 'Linear Regression (Statsmodels OLS)':
        if run:
            result = linear_regression_from_formula(formula, subset_data)
            st.write('### Linear Regression Results (Statsmodels OLS)')

            # R-squared
            r_squared = result.rsquared
            st.write(
                f"The R-squared of the model is {r_squared:.4f}, indicating that the model explains {r_squared * 100:.2f}% of the variance in `{response}`.")

            # Display summary
            st.text(result.summary())

            # Make predictions
            predictions = result.predict(X_mat)

            # Plot actual vs predicted values
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_vec, predictions, color='blue', alpha=0.6)
            min_val, max_val = min(y_vec.min(), predictions.min()), max(y_vec.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted (Linear Regression)')
            ax.legend()
            st.pyplot(fig)

    elif model_type == 'Polynomial Regression (Statsmodels OLS)':
        if run:
            # For polynomial regression, we need a single predictor
            if len(predictor_columns) != 1:
                st.warning("Polynomial regression works best with a single predictor. Using the first predictor only.")
                single_predictor = predictor_columns[0]
            else:
                single_predictor = predictor_columns[0]

            x_data = subset_data[single_predictor].values
            y_data = subset_data[response].values

            result, X_poly = polynomial_regression_from_data(x_data, y_data, degree=degree)

            st.write(f'### Polynomial Regression Results (Degree {degree}, Statsmodels OLS)')

            # R-squared
            r_squared = result.rsquared
            st.write(
                f"The R-squared of the polynomial model is {r_squared:.4f}, indicating that the model explains {r_squared * 100:.2f}% of the variance in `{response}`.")

            # Display summary
            st.text(result.summary())

            # Make predictions
            predictions = result.predict(X_poly)

            # Plot actual vs predicted values
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Actual vs Predicted scatter plot
            ax1.scatter(y_data, predictions, color='blue', alpha=0.6)
            min_val, max_val = min(y_data.min(), predictions.min()), max(y_data.max(), predictions.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            ax1.set_xlabel('Actual')
            ax1.set_ylabel('Predicted')
            ax1.set_title('Actual vs Predicted (Polynomial Regression)')
            ax1.legend()

            # Polynomial fit visualization
            x_sorted_idx = np.argsort(x_data)
            ax2.scatter(x_data, y_data, color='blue', alpha=0.6, label='Data')
            ax2.plot(x_data[x_sorted_idx], predictions[x_sorted_idx], 'r-', label=f'Polynomial Fit (degree {degree})')
            ax2.set_xlabel(single_predictor)
            ax2.set_ylabel(response)
            ax2.set_title(f'Polynomial Fit (Degree {degree})')
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)


    elif model_type == 'Gradient Descent vs OLS Comparison':
        # --- init session state slots once ---
        if 'gd_outputs' not in st.session_state:
            st.session_state['gd_outputs'] = None  # (beta_gd, result_ols, X_for_plotting, history)
        if 'gd_params' not in st.session_state:
            st.session_state['gd_params'] = None  # (lr, epochs)
        if 'gd_show_grad' not in st.session_state:
            st.session_state['gd_show_grad'] = False

        # --- controls inside a FORM so changing them doesn't nuke results until submit ---
        with st.form("gd_form", clear_on_submit=False):
            lr = st.slider('Learning Rate', 0.0005, 0.2, 0.01, step=0.0005, key="gd_lr")
            epochs = st.slider('Epochs', 100, 10000, 1500, step=100, key="gd_epochs")
            show_grad = st.checkbox("Show gradient norm", value=st.session_state['gd_show_grad'], key="gd_show_grad_cb")
            submitted = st.form_submit_button("Run / Update")

        # If submitted or parameters changed, (re)compute and persist
        if submitted or st.session_state['gd_outputs'] is None or st.session_state['gd_params'] != (lr, epochs):
            beta_gd, result_ols, X_for_plotting, history = gradient_descent_with_sm_comparison(
                X_np, y_vec, lr=lr, epochs=epochs
            )
            st.session_state['gd_outputs'] = (beta_gd, result_ols, X_for_plotting, history)
            st.session_state['gd_params'] = (lr, epochs)

        # Persist the checkbox choice (this does NOT trigger recompute)
        st.session_state['gd_show_grad'] = show_grad

        # --- display from persisted outputs ---
        if st.session_state['gd_outputs'] is not None:
            beta_gd, result_ols, X_for_plotting, history = st.session_state['gd_outputs']

            st.subheader("How the optimizer learned")
            fig, ax = plt.subplots()
            ax.plot(history["loss"])
            ax.set_xlabel("Epoch");
            ax.set_ylabel("MSE loss");
            ax.set_title("Training loss over epochs")
            st.pyplot(fig)

            if st.session_state['gd_show_grad']:
                fig2, ax2 = plt.subplots()
                ax2.plot(history["grad_norm"])
                ax2.set_xlabel("Epoch");
                ax2.set_ylabel("‖∇Loss‖");
                ax2.set_title("Gradient norm over epochs")
                st.pyplot(fig2)

            k = min(10, len(history["loss"]))
            snap = pd.DataFrame({"epoch": list(range(k)),
                                 "loss": history["loss"][:k],
                                 "grad_norm": history["grad_norm"][:k]})
            st.write("First epochs snapshot")
            st.dataframe(snap)

            st.subheader("Predictions")
            # X_for_plotting already has the bias column; beta_gd includes intercept
            pred_gd = X_for_plotting @ np.r_[beta_gd[0], beta_gd[1:]]
            pred_ols = result_ols.predict()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.scatter(y_vec, pred_gd, alpha=0.6)
            m1, M1 = float(min(y_vec.min(), pred_gd.min())), float(max(y_vec.max(), pred_gd.max()))
            ax1.plot([m1, M1], [m1, M1], 'r--');
            ax1.set_xlabel('Actual');
            ax1.set_ylabel('Predicted (GD)');
            ax1.set_title('GD Results')

            ax2.scatter(y_vec, pred_ols, alpha=0.6, color='green')
            m2, M2 = float(min(y_vec.min(), pred_ols.min())), float(max(y_vec.max(), pred_ols.max()))
            ax2.plot([m2, M2], [m2, M2], 'r--');
            ax2.set_xlabel('Actual');
            ax2.set_ylabel('Predicted (OLS)');
            ax2.set_title('OLS Results')
            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("Tips for tuning"):
                st.markdown(
                    """
                    - If the loss **oscillates** or **diverges**, reduce the learning rate.  
                    - If learning is **too slow**, try a slightly larger rate.  
                    - **Standardizing features** usually stabilizes learning.  
                    - More epochs won’t help once the loss has **plateaued**.
                    """
                )
