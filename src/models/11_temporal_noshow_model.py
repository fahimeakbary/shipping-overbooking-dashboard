import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "temporal_booking_features.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "temporal_noshow_random_forest_model.pkl"
)

PREDICTION_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "temporal_noshow_prediction_sample.csv"
)

# ==============================
# Load data
# ==============================

df = pd.read_csv(INPUT_PATH)

# ==============================
# Keep only final outcomes
# ==============================

df = df[
    df["Booking Status"].isin(["Loaded", "NoShow"])
].copy()

df["NoShow_Target"] = (
    df["Booking Status"] == "NoShow"
).astype(int)

# ==============================
# Features
# ==============================

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

target_col = "NoShow_Target"

X = df[feature_cols]
y = df[target_col]

categorical_features = [
    "Departure Port",
    "Arrival Port",
    "Sail Time",
    "Ship Code"
]

numeric_features = [
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

# ==============================
# Train / Test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==============================
# Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# ==============================
# Model
# ==============================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=14,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ==============================
# Train
# ==============================

print("=" * 70)
print("TRAINING TEMPORAL NOSHOW MODEL")
print("=" * 70)

pipeline.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ==============================
# Evaluation
# ==============================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)

print("\nROC AUC Score:")
print(round(auc, 4))

# ==============================
# Save model
# ==============================

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

# ==============================
# Save prediction sample
# ==============================

prediction_sample = X_test.copy()
prediction_sample["Actual_NoShow"] = y_test.values
prediction_sample["Predicted_NoShow"] = y_pred
prediction_sample["NoShow_Probability"] = y_proba

prediction_sample.to_csv(PREDICTION_OUTPUT_PATH, index=False)

print("\nPrediction sample saved to:")
print(PREDICTION_OUTPUT_PATH)