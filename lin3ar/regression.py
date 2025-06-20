import numpy as np
from .base import BaseRegressor

class LinearRegression(BaseRegressor):
  def __init__(self, learning_rate=0.01, epochs=1000, solver='gd', lambda_=0):
      self.learning_rate = learning_rate
      self.epochs = epochs
      self.solver = solver
      self.weights = None
      self.lambda_ = lambda_

  def fit(self, X, y):
    y = np.asarray(y).flatten()
    X = np.c_[np.ones((X.shape[0], 1)), X]
    rows, columns = X.shape
    self.weights = np.zeros(columns)
    
    if self.solver == 'gd':
      for _ in range(self.epochs):
        l2_penalty = self.lambda_ * self.weights
        l2_penalty[0] = 0
        
        y_pred = X @ self.weights
        error = y_pred - y
        gradient = (2 / rows) * (X.T @ error) + l2_penalty
        self.weights -= self.learning_rate * gradient
    
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