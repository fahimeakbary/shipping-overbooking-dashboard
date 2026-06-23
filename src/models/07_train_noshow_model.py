import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

BOOKINGS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "shipping_bookings_with_overbooking.csv"
)

CUSTOMER_FEATURES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "customer_features_clean.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "noshow_random_forest_model.pkl"
)

# ==============================
# Load data
# ==============================

bookings = pd.read_csv(BOOKINGS_PATH)
customer_features = pd.read_csv(CUSTOMER_FEATURES_PATH)

# ==============================
# Create target
# ==============================

# 1 = NoShow
# 0 = Loaded
# We only train on final outcomes.
final_bookings = bookings[
    bookings["Booking Status"].isin(["Loaded", "NoShow"])
].copy()

final_bookings["NoShow_Target"] = (
    final_bookings["Booking Status"] == "NoShow"
).astype(int)

# ==============================
# Add customer historical features
# ==============================

model_data = final_bookings.merge(
    customer_features[
        [
            "Customer Number",
            "NoShow_Rate",
            "Booking_Reliability_Score",
            "Meter_Utilization_Rate",
            "Deferral_Candidate_Score"
        ]
    ],
    on="Customer Number",
    how="left"
)

# Fill missing values if any customer feature is missing
model_data = model_data.fillna(0)

# ==============================
# Features and target
# ==============================

feature_cols = [
    "Departure Port",
    "Arrival Port",
    "Sail Time",
    "Ship Code",
    "Reserved Meter",
    "Reserved Heads",
    "NoShow_Rate",
    "Booking_Reliability_Score",
    "Meter_Utilization_Rate",
    "Deferral_Candidate_Score"
]

X = model_data[feature_cols]
y = model_data["NoShow_Target"]

categorical_features = [
    "Departure Port",
    "Arrival Port",
    "Sail Time",
    "Ship Code"
]

numeric_features = [
    "Reserved Meter",
    "Reserved Heads",
    "NoShow_Rate",
    "Booking_Reliability_Score",
    "Meter_Utilization_Rate",
    "Deferral_Candidate_Score"
]

# ==============================
# Train/test split
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
    n_estimators=100,
    max_depth=12,
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
print("TRAINING NOSHOW MODEL")
print("=" * 70)

pipeline.fit(X_train, y_train)

# ==============================
# Predict
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

prediction_sample_path = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "noshow_prediction_sample.csv"
)

prediction_sample.to_csv(prediction_sample_path, index=False)

print("\nPrediction sample saved to:")
print(prediction_sample_path)