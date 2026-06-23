import os
import joblib
import pandas as pd

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "noshow_random_forest_model.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "feature_importance.csv"
)

# ==============================
# Load model
# ==============================

pipeline = joblib.load(MODEL_PATH)

# ==============================
# Extract pieces
# ==============================

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# ==============================
# Feature names
# ==============================

feature_names = preprocessor.get_feature_names_out()

importances = model.feature_importances_

# ==============================
# Create dataframe
# ==============================

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# ==============================
# Save
# ==============================

importance_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==============================
# Print results
# ==============================

print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print("\nTop 20 Features:")

print(
    importance_df.head(20)
)

print("\nOutput saved to:")
print(OUTPUT_PATH)