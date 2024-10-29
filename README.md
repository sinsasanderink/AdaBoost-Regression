# House Sales Prediction using AdaBoost Regression

This project predicts house sale prices and identifies key factors contributing to property values using AdaBoost Regression. Below is an overview of the steps taken, with results from training the model.

## Project Overview

This analysis involves:
1. **Data Loading and Preprocessing**: Import data, rename columns, and check for missing values.
2. **Correlation Analysis**: Identify features highly correlated with house prices.
3. **Model Training**: Train an AdaBoost Regressor model with the most relevant features.
4. **Evaluation**: Assess the model's performance using R² Score and RMSE.

## Key Steps and Results

```python
# Importing Libraries
import numpy as np, pandas as pd, math, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load and preprocess data
data = pd.read_csv('kc_house_data.csv')
df_train = data.copy()
df_train.rename(columns={'price': 'SalePrice'}, inplace=True)
print(f"Missing values: {data.isnull().any().sum()} / {len(data.columns)}")  # Output: 0 / 21 (no missing values)

# Correlation analysis
features = data.iloc[:, 3:].columns.tolist()
correlations = {f"{f} vs price": pearsonr(data[f], data['price'])[0] for f in features}
data_correlations = pd.DataFrame(correlations, index=['Value']).T
print(data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index])

# Selected features and target variable
filtered_data = df_train[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 
                          'view', 'sqft_basement', 'waterfront', 'yr_built', 'lat', 'bedrooms', 'long']]
X = filtered_data.values
y = df_train['SalePrice'].values

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
adaboost = AdaBoostRegressor(n_estimators=50, learning_rate=0.2, loss='exponential').fit(X_train, y_train)

# Model Evaluation
predict = adaboost.predict(X_test)
r2score = r2_score(y_test, predict)
rmse = math.sqrt(mean_squared_error(y_test, predict))
print(f"R² Score: {r2score}, RMSE: {rmse}")

# Example Output
# R² Score: 0.49
# RMSE: $219,185
