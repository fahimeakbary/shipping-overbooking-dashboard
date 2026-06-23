import os
import pandas as pd
import numpy as np

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "shipping_bookings_v3_company_like.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "customer_features_clean.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Load data
# ==============================

bookings = pd.read_csv(RAW_DATA_PATH)

# ==============================
# Keep only finalized bookings
# ==============================
# Booked and Checked In are not final outcomes.
# For customer reliability, we only compare Loaded vs NoShow.

final_statuses = ["Loaded", "NoShow"]

final_bookings = bookings[
    bookings["Booking Status"].isin(final_statuses)
].copy()

# ==============================
# Customer-level features
# ==============================

customer_features = (
    final_bookings
    .groupby("Customer Number")
    .agg(
        Total_Final_Bookings=("Booking Status", "count"),
        Loaded_Bookings=("Booking Status", lambda x: (x == "Loaded").sum()),
        NoShow_Count=("Booking Status", lambda x: (x == "NoShow").sum()),
        Reserved_Meter=("Reserved Meter", "sum"),
        Loaded_Meter=("Loaded Meter", "sum")
    )
    .reset_index()
)

# ==============================
# Main KPIs
# ==============================

customer_features["NoShow_Rate"] = (
    customer_features["NoShow_Count"]
    / customer_features["Total_Final_Bookings"]
)

customer_features["Booking_Reliability_Score"] = (
    customer_features["Loaded_Bookings"]
    / customer_features["Total_Final_Bookings"]
)

customer_features["Unused_Meter"] = (
    customer_features["Reserved_Meter"]
    - customer_features["Loaded_Meter"]
)

customer_features["Meter_Utilization_Rate"] = (
    customer_features["Loaded_Meter"]
    / customer_features["Reserved_Meter"]
)

# ==============================
# Risk category
# ==============================

def assign_risk_category(row):
    reliability = row["Booking_Reliability_Score"]
    noshow_rate = row["NoShow_Rate"]

    if reliability < 0.50 or noshow_rate > 0.50:
        return "High Risk"
    elif reliability < 0.80 or noshow_rate > 0.20:
        return "Medium Risk"
    else:
        return "Low Risk"


customer_features["Risk_Category"] = customer_features.apply(
    assign_risk_category,
    axis=1
)

# ==============================
# Deferral Candidate Logic
# ==============================
# Higher score = better candidate to move to future sailing
# because they have high NoShow behavior and low reliability.

customer_features["Deferral_Candidate_Score"] = (
    0.5 * customer_features["NoShow_Rate"]
    + 0.3 * (1 - customer_features["Booking_Reliability_Score"])
    + 0.2 * customer_features["Meter_Utilization_Rate"].rsub(1)
)

# ==============================
# Sort output
# ==============================

customer_features = customer_features.sort_values(
    by="Deferral_Candidate_Score",
    ascending=False
)

# ==============================
# Save output
# ==============================

customer_features.to_csv(OUTPUT_PATH, index=False)

# ==============================
# Print summary
# ==============================

print("=" * 70)
print("CLEAN CUSTOMER FEATURES CREATED")
print("=" * 70)

print("\nTotal customers:")
print(customer_features["Customer Number"].nunique())

print("\nRisk Category Counts:")
print(customer_features["Risk_Category"].value_counts())

print("\nTop 20 Deferral Candidates:")
print(customer_features.head(20))

print("\nOutput saved to:")
print(OUTPUT_PATH)