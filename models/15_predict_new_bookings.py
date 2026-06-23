import os
import pandas as pd
import joblib

BASE_DIR = r"C:\laptop\shipping_booking_project"

NEW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "new",
    "new_bookings.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "temporal_noshow_random_forest_model.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "predictions",
    "new_booking_predictions.csv"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load new data
new_data = pd.read_csv(NEW_DATA_PATH)

# Load trained model
model = joblib.load(MODEL_PATH)

# Required columns for temporal model
feature_cols = [
    "Departure Port",
    "Arrival Port",
    "Sail Time",
    "Ship Code",
    "Reserved Meter",
    "Reserved Heads",
    "Historical_Bookings",
    "Historical_NoShows",
    "Historical_Loaded",
    "Historical_NoShow_Rate",
    "Historical_Reliability",
    "Historical_Reserved_Meter",
    "Historical_Loaded_Meter",
    "Days_Since_Last_Booking"
]

X_new = new_data[feature_cols]

# Predict
new_data["NoShow_Probability"] = model.predict_proba(X_new)[:, 1]
new_data["Predicted_NoShow"] = model.predict(X_new)

# Risk label
def assign_prediction_risk(prob):
    if prob >= 0.70:
        return "High NoShow Risk"
    elif prob >= 0.40:
        return "Medium NoShow Risk"
    else:
        return "Low NoShow Risk"

new_data["Prediction_Risk_Category"] = new_data["NoShow_Probability"].apply(
    assign_prediction_risk
)

# Save results
new_data.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("NEW BOOKING PREDICTIONS CREATED")
print("=" * 70)

print(new_data[
    [
        "Customer Number",
        "Departure Port",
        "Arrival Port",
        "Sail Date",
        "Sail Time",
        "Ship Code",
        "Reserved Meter",
        "NoShow_Probability",
        "Predicted_NoShow",
        "Prediction_Risk_Category"
    ]
].head(20))

print("\nOutput saved to:")
print(OUTPUT_PATH)