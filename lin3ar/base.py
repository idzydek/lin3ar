import numpy as np
from typing import Dict

class BaseRegressor:
    @staticmethod
    def mean_absolute_error(y: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error (MAE)"""
        return np.mean(np.abs(y - y_pred))

    @staticmethod
    def mean_squared_error(y: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error (MSE)"""
        return np.mean((y - y_pred) ** 2)

    @staticmethod
    def r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score"""
        y_mean = np.mean(y)
        ssr = np.sum((y - y_pred) ** 2)  # Sum of squared residuals
        sst = np.sum((y - y_mean) ** 2)  # Total sum of squares
        if sst == 0:
            return float(0)
        r2 = 1 - (ssr / sst)
        return r2

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model using basic metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
        
        Returns:
            Dictionary containing metric names and their values
        """
        results = {
            'R2 Score': self.r2_score(y_true, y_pred),
            'MAE': self.mean_absolute_error(y_true, y_pred),
            'MSE': self.mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(self.mean_squared_error(y_true, y_pred))
        }

        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        return results