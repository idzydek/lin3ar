import sys
import os
import numpy as np
import pandas as pd
from lin3ar.regression import LinearRegression
from lin3ar.base import BaseRegressor
import pytest

def test_linear_regression_gd():
    """Test: LinearRegression using gradient descent fits simple linear data."""
    X = np.array([[1], [2], [3]])
    y = np.array([3, 5, 7])

    model = LinearRegression(solver='gd', learning_rate=0.1, epochs=1000)
    model.fit(X, y)
    y_pred = model.predict(X)

    model.evaluate_regression(y, y_pred)
    assert np.allclose(y_pred, y, atol=1e-1)


def generate_test_data(n_samples=100, n_features=2, noise=0.1):
    """
    Generate synthetic regression data with a known underlying weight vector.

    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features (excluding bias).
        noise (float): Standard deviation of Gaussian noise added to the target.

    Returns:
        Tuple of (X, y, true_weights)
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features + 1)
    X_with_bias = np.c_[np.ones(n_samples), X]
    y = X_with_bias @ true_weights + noise * np.random.randn(n_samples)
    return X, y, true_weights


class TestBaseRegressor:
    """Unit tests for regression metric calculations in BaseRegressor."""

    def test_metrics_with_perfect_prediction(self):
        """Test: All metrics should return perfect scores for exact predictions."""
        y = np.array([1, 2, 3, 4, 5])
        y_pred = y.copy()

        metrics = BaseRegressor()
        results = metrics.evaluate_regression(y, y_pred)

        assert results['R2 Score'] == 1.0
        assert results['MAE'] == 0.0
        assert results['MSE'] == 0.0
        assert results['RMSE'] == 0.0

    def test_metrics_with_constant_prediction(self):
        """Test: Constant predictions should lead to non-zero errors."""
        y = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])

        metrics = BaseRegressor()
        results = metrics.evaluate_regression(y, y_pred)

        assert results['R2 Score'] == 0.0
        assert results['MAE'] > 0
        assert results['MSE'] > 0
        assert results['RMSE'] > 0


class TestLinearRegression:
    """Unit tests for the LinearRegression class."""

    def test_fit_predict_simple_case(self):
        """Test: Model should learn a simple linear relationship accurately."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])

        model = LinearRegression(learning_rate=0.1, epochs=1000)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert np.allclose(y_pred, y, atol=0.1)
        assert model.weights is not None
        assert len(model.weights) == 2

    def test_fit_predict_multiple_features(self):
        """Test: Model fits multivariate data and estimates correct weights."""
        X, y, true_weights = generate_test_data(n_samples=100, n_features=3)

        model = LinearRegression(learning_rate=0.01, epochs=2000)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert np.allclose(y_pred, y, atol=0.5)
        assert np.allclose(model.weights, true_weights, atol=0.5)

    def test_predict_without_fit(self):
        """Test: predict() should raise an error if called before fit()."""
        model = LinearRegression()
        X = np.array([[1], [2], [3]])

        with pytest.raises(ValueError):
            model.predict(X)

    def test_different_learning_rates(self):
        """Test: Model trains successfully with different learning rates."""
        X, y, _ = generate_test_data(n_samples=50, n_features=2)

        learning_rates = [0.001, 0.01, 0.1]
        for lr in learning_rates:
            model = LinearRegression(learning_rate=lr, epochs=1000)
            model.fit(X, y)
            y_pred = model.predict(X)

            results = model.evaluate_regression(y, y_pred)
            assert results['R2 Score'] > 0.5

    def test_early_stopping(self):
        """Test: With a small learning rate and enough epochs, the model converges."""
        X, y, _ = generate_test_data(n_samples=100, n_features=2)

        model = LinearRegression(learning_rate=0.001, epochs=10000)
        model.fit(X, y)
        y_pred = model.predict(X)

        results = model.evaluate_regression(y, y_pred)
        assert results['R2 Score'] > 0.8

    def test_lambda_work(self):
        """
        Test: Regularization strength (lambda_) should affect the learned weights.
        Checks if weight values change with different regularization.
        """
        X, y, _ = generate_test_data(n_samples=100, n_features=3)

        weights_list = []

        for lam in [0.0, 0.1, 1.0, 10.0]:
            model = LinearRegression(learning_rate=0.01, epochs=1000, lambda_=lam)
            model.fit(pd.DataFrame(X), pd.Series(y))
            weights_list.append(model.weights)

        weights_df = pd.DataFrame(weights_list, columns=["bias", "w1", "w2", "w3"])
        unique_counts = weights_df.nunique()

        assert all(unique_counts > 1)


if __name__ == '__main__':
    pytest.main([__file__])
