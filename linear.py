import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelRF = RandomForestRegressor(n_estimators=100, random_state=42)

modelRF.fit(X_train, y_train)

y_pred_rf = modelRF.predict(X_train)

X_train_stacked = np.column_stack((X_train, y_pred_rf))

X_test_stacked = np.column_stack((X_test, modelRF.predict(X_test)))

modelLR = LinearRegression()

modelLR.fit(X_train_stacked, y_train)

y_pred = modelLR.predict(X_test_stacked)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

plt.scatter(y_test, y_pred, alpha=.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual vs Predicted Prices")
plt.show()