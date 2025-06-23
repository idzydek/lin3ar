import pandas as pd
import numpy as np
from lin3ar.regression import LinearRegression

# Load and preprocess crab age dataset
df = pd.read_csv("./crabs.csv")
df = pd.get_dummies(df, columns=['Plec'], drop_first=True)

df = df[["Plec_I", "Plec_M", "Dlugosc", "Srednica", "Wysokosc", "Waga", "Waga BS", "Waga WN", "Waga SK", "Wiek"]].astype(float)

# Filter rows where total weight equals the sum of component weights
weight_difference = df["Waga"] - (df["Waga BS"] + df["Waga WN"] + df["Waga SK"])
df = df[weight_difference >= 0]

X = df[["Plec_I", "Plec_M", "Dlugosc", "Srednica", "Wysokosc", "Waga", "Waga BS", "Waga WN", "Waga SK"]]
X["Residual"] = weight_difference

y = df["Wiek"].astype(float)

X["Waga BS"] = X["Waga BS"] / X["Waga"]
X["Waga WN"] = X["Waga WN"] / X["Waga"]
X["Waga SK"] = X["Waga SK"] / X["Waga"]
X = pd.DataFrame(X)
X = X.rename(columns={"Waga BS": "BS/Waga", "Waga WN": "WN/Waga", "Waga SK": "SK/Waga"})

model = LinearRegression(solver="ls", lambda_=100)
model.fit(X, y)
ypred = model.predict(X)
model.evaluate_regression(np.array(y), ypred)
