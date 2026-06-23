import os
import pandas as pd

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

OVERBOOKING_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "overbooking_simulation.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "overbooking_recommendations.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Load data
# ==============================

bookings = pd.read_csv(BOOKINGS_PATH)
customers = pd.read_csv(CUSTOMER_FEATURES_PATH)
overbooked_voyages = pd.read_csv(OVERBOOKING_PATH)

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

overbooked_voyages["Voyage_ID"] = (
    overbooked_voyages["Departure Port"].astype(str)
    + "_"
    + overbooked_voyages["Arrival Port"].astype(str)
    + "_"
    + overbooked_voyages["Sail Date"].astype(str)
    + "_"
    + overbooked_voyages["Sail Time"].astype(str)
    + "_"
    + overbooked_voyages["Ship Code"].astype(str)
)

# ==============================
# Attach customer features
# ==============================

bookings_with_customer_score = bookings.merge(
    customers[
        [
            "Customer Number",
            "NoShow_Rate",
            "Booking_Reliability_Score",
            "Meter_Utilization_Rate",
            "Risk_Category",
            "Deferral_Candidate_Score"
        ]
    ],
    on="Customer Number",
    how="left"
)

# ==============================
# Keep only active movable bookings
# ==============================
# We should not recommend NoShow or Cancelled records.
# We mainly move bookings that are still expected/active.

movable_statuses = ["Booked", "Checked In", "Loaded"]

movable_bookings = bookings_with_customer_score[
    bookings_with_customer_score["Booking Status"].isin(movable_statuses)
].copy()

# ==============================
# Recommendation logic
# ==============================

recommendations = []

for _, voyage in overbooked_voyages.iterrows():

    voyage_id = voyage["Voyage_ID"]
    meters_to_remove = voyage["Meters_To_Remove"]

    voyage_bookings = movable_bookings[
        movable_bookings["Voyage_ID"] == voyage_id
    ].copy()

    if voyage_bookings.empty:
        continue

    # Sort candidates:
    # 1. higher Deferral_Candidate_Score first
    # 2. higher NoShow_Rate first
    # 3. lower Reliability first
    voyage_bookings = voyage_bookings.sort_values(
        by=[
            "Deferral_Candidate_Score",
            "NoShow_Rate",
            "Booking_Reliability_Score"
        ],
        ascending=[False, False, True]
    )

    accumulated_meter = 0
    rank = 1

    for _, booking in voyage_bookings.iterrows():

        if accumulated_meter >= meters_to_remove:
            break

        accumulated_meter += booking["Reserved Meter"]

        recommendations.append({
            "Voyage_ID": voyage_id,
            "Departure Port": booking["Departure Port"],
            "Arrival Port": booking["Arrival Port"],
            "Sail Date": booking["Sail Date"],
            "Sail Time": booking["Sail Time"],
            "Ship Code": booking["Ship Code"],
            "Customer Number": booking["Customer Number"],
            "SeqNo": booking["SeqNo"],
            "Booking Status": booking["Booking Status"],
            "Reserved Meter": booking["Reserved Meter"],
            "Meters_To_Remove": meters_to_remove,
            "Accumulated_Removed_Meter": accumulated_meter,
            "Recommendation_Rank": rank,
            "NoShow_Rate": booking["NoShow_Rate"],
            "Booking_Reliability_Score": booking["Booking_Reliability_Score"],
            "Meter_Utilization_Rate": booking["Meter_Utilization_Rate"],
            "Risk_Category": booking["Risk_Category"],
            "Deferral_Candidate_Score": booking["Deferral_Candidate_Score"]
        })

        rank += 1

recommendations_df = pd.DataFrame(recommendations)

# ==============================
# Save output
# ==============================

recommendations_df.to_csv(OUTPUT_PATH, index=False)

# ==============================
# Print summary
# ==============================

print("=" * 70)
print("OVERBOOKING RECOMMENDATION ENGINE")
print("=" * 70)

print("\nTotal overbooked voyages:")
print(len(overbooked_voyages))

print("\nTotal recommendation rows:")
print(len(recommendations_df))

print("\nTop 30 recommendations:")
print(
    recommendations_df[
        [
            "Departure Port",
            "Arrival Port",
            "Sail Date",
            "Sail Time",
            "Ship Code",
            "Customer Number",
            "SeqNo",
            "Booking Status",
            "Reserved Meter",
            "Meters_To_Remove",
            "Accumulated_Removed_Meter",
            "Recommendation_Rank",
            "Risk_Category",
            "Deferral_Candidate_Score"
        ]
    ].head(30)
)

print("\nOutput saved to:")
print(OUTPUT_PATH)