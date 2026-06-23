import os
import pandas as pd

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

VOYAGE_FEATURES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "voyage_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "capacity_utilization.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Load voyage features
# ==============================

voyages = pd.read_csv(VOYAGE_FEATURES_PATH)

# ==============================
# Capacity KPIs
# ==============================

voyages["Capacity_Utilization_Rate"] = (
    voyages["Total_Loaded_Meter"] / voyages["Total_Meter"]
)

voyages["Wasted_Capacity_Meter"] = (
    voyages["Total_Meter"] - voyages["Total_Loaded_Meter"]
)

voyages["Wasted_Capacity_Meter"] = voyages[
    "Wasted_Capacity_Meter"
].clip(lower=0)

voyages["Wasted_Capacity_Rate"] = (
    voyages["Wasted_Capacity_Meter"] / voyages["Total_Meter"]
)

voyages["Overbooking_Ratio"] = (
    voyages["Total_Reserved_Meter"] / voyages["Total_Meter"]
)

voyages["Overbooking_Extra_Meter"] = (
    voyages["Total_Reserved_Meter"] - voyages["Total_Meter"]
)

voyages["Overbooking_Extra_Meter"] = voyages[
    "Overbooking_Extra_Meter"
].clip(lower=0)

voyages["Overbooking_Effectiveness"] = (
    voyages["Total_Loaded_Meter"] / voyages["Total_Reserved_Meter"]
)

# ==============================
# Utilization Category
# ==============================

def assign_utilization_category(rate):
    if rate >= 0.90:
        return "High Utilization"
    elif rate >= 0.70:
        return "Medium Utilization"
    else:
        return "Low Utilization"


voyages["Utilization_Category"] = voyages[
    "Capacity_Utilization_Rate"
].apply(assign_utilization_category)

# ==============================
# Overbooking Outcome
# ==============================

def assign_overbooking_outcome(row):
    overbooked = row["Overbooked_Flag"]
    utilization = row["Capacity_Utilization_Rate"]
    wasted = row["Wasted_Capacity_Rate"]

    if overbooked and utilization >= 0.90:
        return "Successful Overbooking"
    elif overbooked and wasted > 0.20:
        return "Ineffective Overbooking"
    elif overbooked:
        return "Moderate Overbooking"
    else:
        return "Not Overbooked"


voyages["Overbooking_Outcome"] = voyages.apply(
    assign_overbooking_outcome,
    axis=1
)

# ==============================
# Route-level capacity summary
# ==============================

route_capacity = (
    voyages
    .groupby(["Departure Port", "Arrival Port"])
    .agg(
        Total_Voyages=("Ship Code", "count"),
        Avg_Capacity_Utilization=("Capacity_Utilization_Rate", "mean"),
        Avg_Wasted_Capacity=("Wasted_Capacity_Rate", "mean"),
        Avg_Overbooking_Ratio=("Overbooking_Ratio", "mean"),
        Overbooked_Voyages=("Overbooked_Flag", "sum")
    )
    .reset_index()
)

# ==============================
# Save outputs
# ==============================

voyages.to_csv(OUTPUT_PATH, index=False)

route_capacity_path = os.path.join(
    OUTPUT_DIR,
    "route_capacity_summary.csv"
)

route_capacity.to_csv(route_capacity_path, index=False)

# ==============================
# Print summary
# ==============================

print("=" * 70)
print("CAPACITY UTILIZATION ANALYSIS CREATED")
print("=" * 70)

print("\nTotal voyages:")
print(len(voyages))

print("\nAverage Capacity Utilization:")
print(f"{voyages['Capacity_Utilization_Rate'].mean():.2%}")

print("\nAverage Wasted Capacity:")
print(f"{voyages['Wasted_Capacity_Rate'].mean():.2%}")

print("\nUtilization Category Counts:")
print(voyages["Utilization_Category"].value_counts())

print("\nOverbooking Outcome Counts:")
print(voyages["Overbooking_Outcome"].value_counts())

print("\nTop 20 most wasted capacity voyages:")
print(
    voyages
    .sort_values("Wasted_Capacity_Rate", ascending=False)
    [
        [
            "Departure Port",
            "Arrival Port",
            "Sail Date",
            "Sail Time",
            "Ship Code",
            "Total_Meter",
            "Total_Reserved_Meter",
            "Total_Loaded_Meter",
            "Capacity_Utilization_Rate",
            "Wasted_Capacity_Meter",
            "Wasted_Capacity_Rate",
            "Overbooking_Ratio",
            "Overbooking_Outcome"
        ]
    ]
    .head(20)
)

print("\nRoute Capacity Summary:")
print(route_capacity)

print("\nOutput saved to:")
print(OUTPUT_PATH)

print("\nRoute summary saved to:")
print(route_capacity_path)