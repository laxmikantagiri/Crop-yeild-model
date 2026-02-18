import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("data/crop-yield.csv")

# Convert categorical columns to numeric codes
categorical_cols = [
    "Soil_Type",
    "Region",
    "Season",
    "Crop_Type",
    "Irrigation_Type"
]

for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes

# Separate features and target
X = df.drop("Crop_Yield_ton_per_hectare", axis=1).values
y = df["Crop_Yield_ton_per_hectare"].values

# Print feature order (VERY IMPORTANT for inference)
feature_columns = list(df.drop("Crop_Yield_ton_per_hectare", axis=1).columns)

print("\nFeature Order (USE THIS ORDER IN CURL REQUEST):")
for i, col in enumerate(feature_columns):
    print(i, col)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\nRMSE: {rmse:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/pipeline.pkl")

print("\nModel saved successfully for KServe.")

