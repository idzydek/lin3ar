import numpy as np
from .base import BaseRegressor

class LogisticRegression(BaseRegressor):
#   def __init__(self, learning_rate=0.01, epochs=100000, solver='gd'):
#       self.learning_rate = learning_rate
#       self.epochs = epochs
#       self.solver = solver
#       self.weights = None
  
#   def sigmoid(self, z):
#     return 1 / (1 + np.exp(-z))
  
#   def fit(self, X, y):
#     y = y.to_numpy()
#     X = np.c_[np.ones((X.shape[0], 1)), X]
#     rows, columns = X.shape
#     self.weights = np.zeros(columns)
    
#     if self.solver == 'gd':
#       for _ in range(self.epochs):  
#         y_pred = self.sigmoid(X @ self.weights)
#         error = y_pred - y
#         gradient = X.T @ error / rows
#         self.weights -= self.learning_rate * gradient

#     if self.solver == 'sgd':
       

#   def predict(self, X, threshold=0.5):
#     if self.weights is None:
#         raise ValueError("Model not fitted yet!")
#     X = np.c_[np.ones((X.shape[0], 1)), X]
#     y_pred = self.sigmoid(X @ self.weights)
    # return (y_pred >= threshold).astype(int)
    return