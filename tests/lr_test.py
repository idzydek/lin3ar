import sys
import os
import numpy as np
from lin3ar.regression import LinearRegression
from lin3ar.base import BaseRegressor
import pytest

def test_linear_regression_gd():
    X = np.array([[1], [2], [3]])
    y = np.array([3, 5, 7])

    model = LinearRegression(solver='gd', learning_rate=0.1, epochs=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    model.evaluate_regression(y, y_pred)

    assert np.allclose(y_pred, y, atol=1e-1)

def generate_test_data(n_samples=100, n_features=2, noise=0.1):
    """Generate synthetic test data"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features + 1)
    X_with_bias = np.c_[np.ones(n_samples), X]
    y = X_with_bias @ true_weights + noise * np.random.randn(n_samples)
    return X, y, true_weights

class TestBaseRegressor:
    def test_metrics_with_perfect_prediction(self):
        y = np.array([1, 2, 3, 4, 5])
        y_pred = y.copy()
        
        metrics = BaseRegressor()
        results = metrics.evaluate_regression(y, y_pred)
        
        assert results['R2 Score'] == 1.0
        assert results['MAE'] == 0.0
        assert results['MSE'] == 0.0
        assert results['RMSE'] == 0.0

    def test_metrics_with_constant_prediction(self):
        y = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])
        
        metrics = BaseRegressor()
        results = metrics.evaluate_regression(y, y_pred)
        
        assert results['R2 Score'] == 0.0
        assert results['MAE'] > 0
        assert results['MSE'] > 0
        assert results['RMSE'] > 0

class TestLinearRegression:
    def test_fit_predict_simple_case(self):
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model = LinearRegression(learning_rate=0.1, epochs=1000)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert np.allclose(y_pred, y, atol=0.1)
        assert model.weights is not None
        assert len(model.weights) == 2

    def test_fit_predict_multiple_features(self):
        X, y, true_weights = generate_test_data(n_samples=100, n_features=3)
        
        model = LinearRegression(learning_rate=0.01, epochs=2000)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Check if predictions are close to true values
        assert np.allclose(y_pred, y, atol=0.5)
        
        # Check if learned weights are close to true weights
        assert np.allclose(model.weights, true_weights, atol=0.5)

    def test_predict_without_fit(self):
        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        
        with pytest.raises(ValueError):
            model.predict(X)

    def test_different_learning_rates(self):
        X, y, _ = generate_test_data(n_samples=50, n_features=2)
        
        # Test with different learning rates
        learning_rates = [0.001, 0.01, 0.1]
        for lr in learning_rates:
            model = LinearRegression(learning_rate=lr, epochs=1000)
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Check if model converges
            results = model.evaluate_regression(y, y_pred)
            assert results['R2 Score'] > 0.5

    def test_early_stopping(self):
        X, y, _ = generate_test_data(n_samples=100, n_features=2)
        
        # Test with very small learning rate to ensure convergence
        model = LinearRegression(learning_rate=0.001, epochs=10000)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Check if model achieves good performance
        results = model.evaluate_regression(y, y_pred)
        assert results['R2 Score'] > 0.8

if __name__ == '__main__':
    pytest.main([__file__])