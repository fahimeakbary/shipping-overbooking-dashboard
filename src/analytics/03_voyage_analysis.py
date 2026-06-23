import os
import pandas as pd

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "shipping_bookings_with_overbooking.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "voyage_features.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Load data
# ==============================

bookings = pd.read_csv(RAW_DATA_PATH)

# ==============================
# Create Voyage ID
# ==============================

voyage_cols = [
    "Departure Port",
    "Arrival Port",
    "Sail Date",
    "Sail Time",
    "Ship Code"
]

bookings["Voyage_ID"] = (
    bookings["Departure Port"].astype(str)
    + "_"
    + bookings["Arrival Port"].astype(str)
    + "_"
    + bookings["Sail Date"].astype(str)
    + "_"
    + bookings["Sail Time"].astype(str)
    + "_"
    + bookings["Ship Code"].astype(str)
)

# ==============================
# Voyage-level aggregation
# ==============================

voyage_features = (
    bookings
    .groupby(voyage_cols)
    .agg(
        Total_Bookings=("Booking Status", "count"),
        Loaded_Bookings=("Booking Status", lambda x: (x == "Loaded").sum()),
        NoShow_Count=("Booking Status", lambda x: (x == "NoShow").sum()),
        Booked_Count=("Booking Status", lambda x: (x == "Booked").sum()),
        CheckedIn_Count=("Booking Status", lambda x: (x == "Checked In").sum()),
        Cancelled_Count=("Booking Status", lambda x: (x == "Cancelled").sum()),
        Total_Reserved_Meter=("Reserved Meter", "sum"),
        Total_Loaded_Meter=("Loaded Meter", "sum"),
        Total_Reserved_Heads=("Reserved Heads", "sum"),
        Total_Loaded_Heads=("Loaded Heads", "sum"),
        Total_Meter=("Total Meter", "first"),
        Total_Head=("Total Head", "first")
    )
    .reset_index()
)

# ==============================
# Voyage KPIs
# ==============================

voyage_features["NoShow_Rate"] = (
    voyage_features["NoShow_Count"]
    / voyage_features["Total_Bookings"]
)

voyage_features["Capacity_Utilization_Rate"] = (
    voyage_features["Total_Loaded_Meter"]
    / voyage_features["Total_Meter"]
)

voyage_features["Reservation_Ratio"] = (
    voyage_features["Total_Reserved_Meter"]
    / voyage_features["Total_Meter"]
)

voyage_features["Overbooked_Flag"] = (
    voyage_features["Total_Reserved_Meter"]
    > voyage_features["Total_Meter"]
)

voyage_features["Overbooked_Meter"] = (
    voyage_features["Total_Reserved_Meter"]
    - voyage_features["Total_Meter"]
)

voyage_features["Overbooked_Meter"] = voyage_features[
    "Overbooked_Meter"
].clip(lower=0)

voyage_features["Unused_Capacity_Meter"] = (
    voyage_features["Total_Meter"]
    - voyage_features["Total_Loaded_Meter"]
)

voyage_features["Unused_Capacity_Meter"] = voyage_features[
    "Unused_Capacity_Meter"
].clip(lower=0)

# ==============================
# Risk category for voyage
# ==============================

def assign_voyage_risk(row):
    reservation_ratio = row["Reservation_Ratio"]
    noshow_rate = row["NoShow_Rate"]

    if reservation_ratio > 1.10 and noshow_rate < 0.15:
        return "High Overbooking Risk"
    elif reservation_ratio > 1.00:
        return "Overbooked"
    elif noshow_rate > 0.25:
        return "High NoShow Risk"
    else:
        return "Normal"


voyage_features["Voyage_Risk_Category"] = voyage_features.apply(
    assign_voyage_risk,
    axis=1
)

# ==============================
# Sort output
# ==============================

voyage_features = voyage_features.sort_values(
    by=["Overbooked_Flag", "Reservation_Ratio"],
    ascending=[False, False]
)

# ==============================
# Save output
# ==============================

voyage_features.to_csv(OUTPUT_PATH, index=False)

# ==============================
# Print summary
# ==============================

print("=" * 70)
print("VOYAGE FEATURES CREATED")
print("=" * 70)

print("\nTotal voyages:")
print(len(voyage_features))

print("\nOverbooked voyages:")
print(voyage_features["Overbooked_Flag"].sum())

print("\nVoyage Risk Category Counts:")
print(voyage_features["Voyage_Risk_Category"].value_counts())

print("\nTop 20 voyages by Reservation Ratio:")
print(
    voyage_features[
        [
            "Departure Port",
            "Arrival Port",
            "Sail Date",
            "Sail Time",
            "Ship Code",
            "Total_Bookings",
            "Total_Reserved_Meter",
            "Total_Loaded_Meter",
            "Total_Meter",
            "Reservation_Ratio",
            "Capacity_Utilization_Rate",
            "NoShow_Rate",
            "Overbooked_Flag",
            "Overbooked_Meter",
            "Voyage_Risk_Category"
        ]
    ].head(20)
)

print("\nOutput saved to:")
print(OUTPUT_PATH)