# Lin3ar - Linear Regression Library

A lightweight, educational Python library for linear regression with gradient descent and least squares solvers. Built from scratch using NumPy for learning and experimentation purposes.

## 🎯 Overview

Lin3ar is a simple yet powerful linear regression implementation that provides:
- **Gradient Descent** solver with customizable learning rate and epochs
- **Least Squares** solver using matrix operations
- **L2 Regularization** support to prevent overfitting
- **Comprehensive evaluation metrics** (R², MAE, MSE, RMSE)
- **Data preprocessing** utilities
- **Extensive test suite** with synthetic data generation

## 📁 Project Structure

```
lin3ar/
├── lin3ar/                    # Main package
│   ├── __init__.py           # Package exports
│   ├── base.py               # Base regressor with evaluation metrics
│   ├── regression.py         # Linear regression implementation
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── logistic.py           # Logistic regression (work in progress)
├── tests/                    # Test suite
│   ├── __init__.py
│   └── lr_test.py           # Comprehensive unit tests
├── notebooks/                # Jupyter notebooks with examples
│   └── linear_regression/
│       └── crabs/           # Real-world crab age prediction example
├── pyproject.toml           # Build configuration
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lin3ar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## 📖 Usage

### Basic Linear Regression

```python
import numpy as np
from lin3ar.regression import LinearRegression

# Generate sample data
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
y = np.array([2, 4, 6, 8])

# Create and fit model
model = LinearRegression(learning_rate=0.01, epochs=1000, solver='gd')
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate model
model.evaluate_regression(y, predictions)
```

### Using Least Squares Solver

```python
# For faster convergence with smaller datasets
model = LinearRegression(solver='ls')
model.fit(X, y)
```

### With L2 Regularization

```python
# Add regularization to prevent overfitting
model = LinearRegression(solver='gd', lambda_=0.1, learning_rate=0.01, epochs=1000)
model.fit(X, y)
```

### Data Preprocessing

```python
from lin3ar.preprocessing import standard_scale

# Standardize features
X_scaled = standard_scale(X)
```

## 🔧 Features

### Linear Regression Class

The `LinearRegression` class supports:

- **Multiple solvers**: Gradient descent (`'gd'`) and least squares (`'ls'`)
- **Hyperparameter tuning**: Learning rate, epochs, regularization strength
- **Automatic bias term**: Adds intercept automatically
- **Robust error handling**: Handles singular matrices gracefully

### Evaluation Metrics

The `BaseRegressor` class provides comprehensive evaluation:

- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error

### Data Preprocessing

- **Standard scaling**: Zero mean and unit variance normalization
- **Pandas compatibility**: Works with both NumPy arrays and Pandas DataFrames

## 🧪 Testing

Run the comprehensive test suite:

```bash
pytest tests/
```

The test suite includes:
- Unit tests for all evaluation metrics
- Synthetic data generation for testing
- Gradient descent convergence tests
- Regularization effectiveness tests
- Error handling validation

## 📊 Real-World Example

The project includes a practical example using crab age prediction:

```python
# See notebooks/linear_regression/crabs/regression_demo.py
import pandas as pd
from lin3ar.regression import LinearRegression

# Load crab dataset
df = pd.read_csv("crabs.csv")

# Preprocess data
df = pd.get_dummies(df, columns=['Plec'], drop_first=True)
X = df[["Plec_I", "Plec_M", "Dlugosc", "Srednica", "Wysokosc", "Waga"]]
y = df["Wiek"]

# Train model with regularization
model = LinearRegression(solver="ls", lambda_=100)
model.fit(X, y)
predictions = model.predict(X)

# Evaluate performance
model.evaluate_regression(y, predictions)
```

## 🛠️ Dependencies

- **NumPy**: Core numerical computations
- **Pytest**: Testing framework
- **Pandas**: Data manipulation (optional, for preprocessing)
- **Matplotlib/Seaborn**: Visualization (optional, for notebooks)

## 🎓 Educational Value

This library is designed for educational purposes and demonstrates:

- **Algorithm implementation**: Gradient descent and least squares from scratch
- **Software engineering**: Clean code structure, testing, and documentation
- **Machine learning concepts**: Regularization, evaluation metrics, data preprocessing
- **Python best practices**: Type hints, error handling, modular design

## 🔮 Future Enhancements

- [ ] Complete logistic regression implementation
- [ ] Stochastic gradient descent
- [ ] Cross-validation utilities
- [ ] More preprocessing options
- [ ] Model persistence (save/load)
- [ ] Visualization tools

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Note**: This library is primarily for educational purposes. For production use, consider established libraries like scikit-learn.
