# ============================================================
# Soil Moisture Prediction - Model Training Script
# ============================================================
# This script loads the soil moisture dataset, preprocesses the data,
# trains a machine learning model, evaluates it, and finally saves
# the trained model and scaler for use in the Flask web application.
# ============================================================

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# ------------------------------------------------------------
# STEP 1: Ensure 'models/' folder exists to store saved files
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------
# STEP 2: Load dataset
# ------------------------------------------------------------
# We are using the soil moisture dataset stored in 'data/soil-moisture.csv'.
# The dataset has multiple sensor readings (PM, humidity, temp, etc.)
# and the target column 'avg_sm' which represents soil moisture.
df = pd.read_csv("data/soil-moisture.csv")

# ------------------------------------------------------------
# STEP 3: Define Features (X) and Target (y)
# ------------------------------------------------------------
# Ignoring 'Month' and 'Day' for simplicity.
# Features: Sensor readings like PM1, PM2, temperature, humidity, etc.
# Target: Soil moisture ('avg_sm').
X = df[['avg_pm1','avg_pm2','avg_pm3','avg_am','avg_lum','avg_temp','avg_humd','avg_pres']]
y = df['avg_sm']

# ------------------------------------------------------------
# STEP 4: Split dataset into Training and Testing sets
# ------------------------------------------------------------
# 80% data for training, 20% for testing (validation).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# STEP 5: Standardize (Scale) the data
# ------------------------------------------------------------
# Scaling helps models like RandomForest to perform consistently
# by normalizing feature ranges.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------------------------------------------
# STEP 6: Train the Model
# ------------------------------------------------------------
# We use Random Forest Regressor because it handles non-linear data well
# and is robust to noise. Using 200 trees for stable performance.
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_s, y_train)

# ------------------------------------------------------------
# STEP 7: Make Predictions on Test Data
# ------------------------------------------------------------
y_pred = model.predict(X_test_s)

# ------------------------------------------------------------
# STEP 8: Save Model and Scaler
# ------------------------------------------------------------
# Save the trained RandomForest model and the scaler as .pkl files.
# These will later be loaded in the Flask app for real-time predictions.
joblib.dump(model, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ------------------------------------------------------------
# STEP 9: Evaluate the Model
# ------------------------------------------------------------
# Use standard regression metrics:
# - R² Score (goodness of fit, closer to 1 is better)
# - MAE (Mean Absolute Error, lower is better)
# - RMSE (Root Mean Squared Error, lower is better)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ------------------------------------------------------------
# STEP 10: Save Metrics for Reference
# ------------------------------------------------------------
# Store evaluation results in 'metrics.txt' inside models/ folder
with open("models/metrics.txt", "w") as f:
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")

# ------------------------------------------------------------
# STEP 11: Print Success Message
# ------------------------------------------------------------
print("✅ Model and scaler saved in /models/")
print(f"R² Score: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
