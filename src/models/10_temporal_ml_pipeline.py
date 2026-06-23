import os
import pandas as pd
import numpy as np

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "shipping_bookings_with_overbooking.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "temporal_booking_features.csv"
)

# ==============================
# Load data
# ==============================

df = pd.read_csv(INPUT_PATH)

# ==============================
# Create datetime
# ==============================

df["Sail_Datetime"] = pd.to_datetime(
    df["Sail Date"] + " " + df["Sail Time"]
)

# ==============================
# Sort
# ==============================

df = df.sort_values(
    ["Customer Number", "Sail_Datetime"]
)

# ==============================
# Target helpers
# ==============================

df["Is_NoShow"] = (
    df["Booking Status"] == "NoShow"
).astype(int)

df["Is_Loaded"] = (
    df["Booking Status"] == "Loaded"
).astype(int)

# ==============================
# Historical Features
# ==============================

customer_groups = []

for customer, group in df.groupby("Customer Number"):

    group = group.copy()

    # Number of previous bookings
    group["Historical_Bookings"] = np.arange(len(group))

    # Previous NoShows
    group["Historical_NoShows"] = (
        group["Is_NoShow"]
        .shift()
        .fillna(0)
        .cumsum()
    )

    # Previous Loaded
    group["Historical_Loaded"] = (
        group["Is_Loaded"]
        .shift()
        .fillna(0)
        .cumsum()
    )

    # Historical NoShow Rate
    group["Historical_NoShow_Rate"] = (
        group["Historical_NoShows"]
        /
        group["Historical_Bookings"].replace(0, np.nan)
    )

    # Historical Reliability
    group["Historical_Reliability"] = (
        group["Historical_Loaded"]
        /
        group["Historical_Bookings"].replace(0, np.nan)
    )

    # Historical Reserved Meter
    group["Historical_Reserved_Meter"] = (
        group["Reserved Meter"]
        .shift()
        .fillna(0)
        .cumsum()
    )

    # Historical Loaded Meter
    group["Historical_Loaded_Meter"] = (
        group["Loaded Meter"]
        .shift()
        .fillna(0)
        .cumsum()
    )

    # Days since previous booking
    group["Days_Since_Last_Booking"] = (
        group["Sail_Datetime"]
        -
        group["Sail_Datetime"].shift()
    ).dt.days

    customer_groups.append(group)

# ==============================
# Combine
# ==============================

temporal_df = pd.concat(customer_groups)

# ==============================
# Fill first bookings
# ==============================

temporal_df = temporal_df.fillna(0)

# ==============================
# Save
# ==============================

temporal_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==============================
# Summary
# ==============================

print("=" * 70)
print("TEMPORAL FEATURES CREATED")
print("=" * 70)

print("\nShape:")
print(temporal_df.shape)

print("\nColumns created:")

new_cols = [
    "Historical_Bookings",
    "Historical_NoShows",
    "Historical_Loaded",
    "Historical_NoShow_Rate",
    "Historical_Reliability",
    "Historical_Reserved_Meter",
    "Historical_Loaded_Meter",
    "Days_Since_Last_Booking"
]

for col in new_cols:
    print(col)

print("\nOutput saved to:")
print(OUTPUT_PATH)