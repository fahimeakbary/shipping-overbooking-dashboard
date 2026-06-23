import os
import pandas as pd

BASE_DIR = r"C:\laptop\shipping_booking_project"

FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
os.makedirs(DASHBOARD_DIR, exist_ok=True)

capacity_path = os.path.join(FEATURE_DIR, "capacity_utilization.csv")
customer_path = os.path.join(FEATURE_DIR, "customer_features_clean.csv")
recommendation_path = os.path.join(FEATURE_DIR, "smart_overbooking_recommendations.csv")
prediction_path = os.path.join(FEATURE_DIR, "temporal_noshow_prediction_sample.csv")

capacity = pd.read_csv(capacity_path)
customers = pd.read_csv(customer_path)
recommendations = pd.read_csv(recommendation_path)
predictions = pd.read_csv(prediction_path)

# ==============================
# KPI Summary
# ==============================

kpi_summary = pd.DataFrame({
    "Metric": [
        "Total Voyages",
        "Average Capacity Utilization",
        "Average Wasted Capacity",
        "Overbooked Voyages",
        "High Risk Customers",
        "Total Recommendations",
        "Average NoShow Probability"
    ],
    "Value": [
        len(capacity),
        round(capacity["Capacity_Utilization_Rate"].mean(), 4),
        round(capacity["Wasted_Capacity_Rate"].mean(), 4),
        int(capacity["Overbooked_Flag"].sum()),
        int((customers["Risk_Category"] == "High Risk").sum()),
        len(recommendations),
        round(predictions["NoShow_Probability"].mean(), 4)
    ]
})

# ==============================
# Top Risky Customers
# ==============================

top_risky_customers = customers.sort_values(
    by="Deferral_Candidate_Score",
    ascending=False
).head(50)

# ==============================
# Top Overbooked Voyages
# ==============================

top_overbooked_voyages = capacity.sort_values(
    by="Overbooking_Extra_Meter",
    ascending=False
).head(50)

# ==============================
# Top Smart Recommendations
# ==============================

top_recommendations = recommendations.sort_values(
    by="Smart_Deferral_Score",
    ascending=False
).head(100)

# ==============================
# Save Dashboard Tables
# ==============================

kpi_summary.to_csv(
    os.path.join(DASHBOARD_DIR, "kpi_summary.csv"),
    index=False
)

top_risky_customers.to_csv(
    os.path.join(DASHBOARD_DIR, "top_risky_customers.csv"),
    index=False
)

top_overbooked_voyages.to_csv(
    os.path.join(DASHBOARD_DIR, "top_overbooked_voyages.csv"),
    index=False
)

top_recommendations.to_csv(
    os.path.join(DASHBOARD_DIR, "top_smart_recommendations.csv"),
    index=False
)

print("=" * 70)
print("DASHBOARD DATASETS CREATED")
print("=" * 70)

print("\nKPI Summary:")
print(kpi_summary)

print("\nFiles saved in:")
print(DASHBOARD_DIR)