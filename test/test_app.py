import unittest
import numpy as np
import pandas as pd
from statsmodels.api import OLS
from patsy import dmatrices

# Import the custom classes and functions from the script
from src.app_functions.modeling_from_user import DecisionTree, RandomForest, linear_regression_from_formula

class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        # Setup dummy data for DecisionTree tests
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.y = np.array([2, 3, 4, 5, 6])
        self.tree = DecisionTree(max_depth=3, min_samples_split=2)

    def test_fit(self):
        # Test if the tree can be fitted without errors
        self.tree.tree = self.tree.fit(self.X, self.y)
        self.assertIsNotNone(self.tree.tree)

    def test_predict(self):
        # Test if predictions return a valid output
        self.tree.tree = self.tree.fit(self.X, self.y)
        predictions = self.tree.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))


class TestRandomForest(unittest.TestCase):

    def setUp(self):
        # Setup dummy data for RandomForest tests
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.y = np.array([2, 3, 4, 5, 6])
        self.rf = RandomForest(n_estimators=5, max_depth=3, min_samples_split=2)

    def test_fit(self):
        # Test if the random forest can be fitted without errors
        self.rf.fit(self.X, self.y)
        self.assertEqual(len(self.rf.trees), 5)

    def test_predict(self):
        # Test if random forest predictions are generated correctly
        self.rf.fit(self.X, self.y)
        predictions = self.rf.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))


class TestLinearRegressionFromFormula(unittest.TestCase):

    def setUp(self):
        # Create a dummy DataFrame for testing linear regression
        self.data = pd.DataFrame({
            'soil_organic_carbon': [1.2, 2.3, 3.1, 4.5, 5.1],
            'aspect': [0.1, 0.2, 0.3, 0.4, 0.5],
            'bd_mean': [10, 15, 20, 25, 30]
        })
        self.formula = 'soil_organic_carbon ~ aspect + bd_mean'

    def test_linear_regression(self):
        # Test if linear regression runs successfully
        result = linear_regression_from_formula(self.formula, self.data)
        self.assertIsInstance(result, OLS)

    def test_regression_predictions(self):
        # Test if the regression produces predictions
        result = linear_regression_from_formula(self.formula, self.data)
        y, X = dmatrices(self.formula, self.data)
        predictions = result.predict(X)
        self.assertEqual(len(predictions), len(self.data))


if __name__ == '__main__':
    unittest.main()
