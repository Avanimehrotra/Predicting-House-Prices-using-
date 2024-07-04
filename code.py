import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('house_prices_dataset.csv')

# Handle missing values if any
data = data.fillna(0)  # Replace missing values with 0, you can use other strategies for handling missing data

# Encode categorical variables (e.g., location) using one-hot encoding
data = pd.get_dummies(data, columns=['location'])

# Define features (X) and target variable (Y)
X = data[['bedrooms', 'bathrooms', 'square_footage', 'house_age']]

# For linear regression (single feature example)
X_single = data[['square_footage']]
Y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, Y_train)

# Predict house prices
Y_pred = model.predict(X_test)

# Model evaluation for linear regression
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Linear Regression Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs. predicted prices for linear regression
plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices (Linear Regression)")
plt.show()

# Multiple Regression model
model_multi = LinearRegression()

# Fit the model to the training data
model_multi.fit(X_train, Y_train)

# Predict house prices
Y_pred_multi = model_multi.predict(X_test)

# Model evaluation for multiple regression
mae_multi = mean_absolute_error(Y_test, Y_pred_multi)
mse_multi = mean_squared_error(Y_test, Y_pred_multi)
rmse_multi = np.sqrt(mse_multi)

print("\nMultiple Regression Metrics:")
print(f"Mean Absolute Error (MAE): {mae_multi}")
print(f"Mean Squared Error (MSE): {mse_multi}")
print(f"Root Mean Squared Error (RMSE): {rmse_multi}")

# Plot actual vs. predicted prices for multiple regression
plt.scatter(Y_test, Y_pred_multi)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices (Multiple Regression)")
plt.show()

