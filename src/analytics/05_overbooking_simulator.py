import os
import pandas as pd

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

VOYAGE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "voyage_features.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "overbooking_simulation.csv"
)

# ==============================
# Load Data
# ==============================

voyages = pd.read_csv(VOYAGE_PATH)

# ==============================
# Keep only overbooked voyages
# ==============================

overbooked = voyages[
    voyages["Overbooked_Flag"] == True
].copy()

# ==============================
# Simulation
# ==============================

overbooked["Meters_To_Remove"] = (
    overbooked["Overbooked_Meter"]
)

# میانگین هر booking حدود 14 متر بود
AVERAGE_BOOKING_SIZE = 14

overbooked["Estimated_Bookings_To_Move"] = (
    overbooked["Meters_To_Remove"]
    / AVERAGE_BOOKING_SIZE
).round().astype(int)

# ==============================
# Severity
# ==============================

def assign_severity(extra_meter):

    if extra_meter < 100:
        return "Low"

    elif extra_meter < 300:
        return "Medium"

    elif extra_meter < 600:
        return "High"

    else:
        return "Critical"


overbooked["Overbooking_Severity"] = (
    overbooked["Overbooked_Meter"]
    .apply(assign_severity)
)

# ==============================
# Sort
# ==============================

overbooked = overbooked.sort_values(
    by="Overbooked_Meter",
    ascending=False
)

# ==============================
# Save
# ==============================

overbooked.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==============================
# Print Results
# ==============================

print("=" * 70)
print("OVERBOOKING SIMULATION")
print("=" * 70)

print("\nTotal Overbooked Voyages:")
print(len(overbooked))

print("\nSeverity Counts:")
print(
    overbooked["Overbooking_Severity"]
    .value_counts()
)

print("\nAverage Meters To Remove:")
print(
    round(
        overbooked["Meters_To_Remove"].mean(),
        2
    )
)

print("\nTop 20 Critical Voyages:")

print(
    overbooked[
        [
            "Departure Port",
            "Arrival Port",
            "Sail Date",
            "Ship Code",
            "Overbooked_Meter",
            "Meters_To_Remove",
            "Estimated_Bookings_To_Move",
            "Overbooking_Severity"
        ]
    ]
    .head(20)
)

print("\nOutput saved to:")
print(OUTPUT_PATH)