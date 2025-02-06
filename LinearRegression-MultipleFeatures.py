import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
df = pd.read_csv(url)

# Check dataset structure
print(df.head())
print(df.isnull().sum())

# Features to analyze
selected_features = ['rm', 'lstat', 'ptratio', 'indus', 'tax']
metrics = {
    'Feature': [],
    'MSE': [],
    'RMSE': [],
    'R-squared': []
}

# Function to analyze a single feature
def analyzed_feature(feature):
    X = df[[feature]]
    y = df['medv']  # Target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = lr_model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store results
    metrics['Feature'].append(feature)
    metrics['MSE'].append(mse)
    metrics['RMSE'].append(rmse)
    metrics['R-squared'].append(r2)

    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={"s": 60}, line_kws={"color": "red"})
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Actual vs Predicted House Prices ({feature})")
    plt.show()

# Analyze each feature
for feature in selected_features:
    analyzed_feature(feature)

# Convert metrics dictionary to DataFrame
metrics_df = pd.DataFrame(metrics)

# Plot MSE, RMSE, and R-squared comparisons
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# MSE Comparison
sns.barplot(x='Feature', y='MSE', data=metrics_df, ax=ax[0])
ax[0].set_title('MSE Comparison')
ax[0].set_xticklabels(metrics_df['Feature'], rotation=45)

# RMSE Comparison
sns.barplot(x='Feature', y='RMSE', data=metrics_df, ax=ax[1])
ax[1].set_title('RMSE Comparison')
ax[1].set_xticklabels(metrics_df['Feature'], rotation=45)

# R-squared Comparison
sns.barplot(x='Feature', y='R-squared', data=metrics_df, ax=ax[2])
ax[2].set_title('R-squared Comparison')
ax[2].set_xticklabels(metrics_df['Feature'], rotation=45)

plt.tight_layout()
plt.show()
