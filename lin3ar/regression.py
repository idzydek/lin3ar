import numpy as np
from .base import BaseRegressor

class LinearRegression(BaseRegressor):
  def __init__(self, learning_rate=0.01, epochs=1000, solver='gd', lambda_=0, penalty='l2'):
      self.penalty = penalty
      self.learning_rate = learning_rate
      self.epochs = epochs
      self.solver = solver
      self.weights = None
      self.lambda_ = lambda_

      if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
      if epochs <= 0:
        raise ValueError("Number of epochs must be positive.")
      if solver not in ['gd', 'ls']:
        raise ValueError("Solver must be either 'gd' (gradient descent) or 'ls' (least squares).")
      if lambda_ < 0:
        raise ValueError("Regularization parameter lambda must be non-negative.")
      if penalty not in ['l2', 'l1']:
        raise ValueError("Penalty must be either 'l2' or 'l1'.")
  
  def fit(self, X, y):
    """
    Fit the linear regression model using gradient descent.

    Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features)
        y (array-like): Target vector of shape (n_samples,)
    """
    
    if not isinstance(X, np.ndarray):
      X = np.asarray(X)
    if not isinstance(y, np.ndarray):
      y = np.asarray(y)

    X = np.c_[np.ones((X.shape[0], 1)), X] # Add bias term
    if X.ndim != 2 or y.ndim != 1:
      raise ValueError("X must be a 2D array and y must be a 1D array.")
    
    rows, columns = X.shape
    self.weights = np.zeros(columns)
    
    if self.solver == 'gd':
      for _ in range(self.epochs):
        if self.penalty == 'l1':
          regularization_term = np.sign(self.weights) * self.lambda_
        elif self.penalty == 'l2':
          regularization_term = 2 * self.lambda_ * self.weights
        else:
          regularization_term = 0
        regularization_term[0] = 0

        y_pred = X @ self.weights
        error = y_pred - y
        gradient = (2 / rows) * (X.T @ error)
        self.weights -= self.learning_rate * (gradient + regularization_term)
    
    if self.solver == 'ls':
      try:
        self.weights = np.linalg.solve(X.T @ X, X.T @ y)
      except np.linalg.LinAlgError:
        self.weights = np.linalg.pinv(X) @ y

  def predict(self, X):
    if self.weights is None:
      raise ValueError("Model not fitted yet!")
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X @ self.weights 