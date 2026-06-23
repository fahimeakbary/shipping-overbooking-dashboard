import os
import pandas as pd

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"

FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
POWERBI_DIR = os.path.join(BASE_DIR, "dashboard", "powerbi_exports")

os.makedirs(POWERBI_DIR, exist_ok=True)

# ==============================
# Input files
# ==============================

files_to_export = {
    "customer_features": os.path.join(FEATURE_DIR, "customer_features_clean.csv"),
    "voyage_features": os.path.join(FEATURE_DIR, "voyage_features.csv"),
    "capacity_utilization": os.path.join(FEATURE_DIR, "capacity_utilization.csv"),
    "overbooking_simulation": os.path.join(FEATURE_DIR, "overbooking_simulation.csv"),
    "smart_recommendations": os.path.join(FEATURE_DIR, "smart_overbooking_recommendations.csv"),
    "feature_importance": os.path.join(FEATURE_DIR, "feature_importance.csv"),
    "kpi_summary": os.path.join(DASHBOARD_DIR, "kpi_summary.csv"),
    "top_risky_customers": os.path.join(DASHBOARD_DIR, "top_risky_customers.csv"),
    "top_overbooked_voyages": os.path.join(DASHBOARD_DIR, "top_overbooked_voyages.csv"),
    "top_smart_recommendations": os.path.join(DASHBOARD_DIR, "top_smart_recommendations.csv"),
}

# ==============================
# Export cleaned CSV files
# ==============================

print("=" * 70)
print("POWER BI EXPORT")
print("=" * 70)

for name, path in files_to_export.items():
    if not os.path.exists(path):
        print(f"Missing file, skipped: {path}")
        continue

    df = pd.read_csv(path)

    # Clean column names for Power BI
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    output_path = os.path.join(
        POWERBI_DIR,
        f"{name}.csv"
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Exported: {output_path} | Shape: {df.shape}")

# ==============================
# Create Power BI README
# ==============================

readme_path = os.path.join(POWERBI_DIR, "README_powerbi.txt")

readme_text = """
Power BI Export Files

Use these CSV files in Power BI Desktop:

1. customer_features.csv
   Customer-level reliability, no-show rate, risk category and deferral score.

2. voyage_features.csv
   Voyage-level booking, no-show, reservation and capacity information.

3. capacity_utilization.csv
   Capacity utilization, wasted capacity and overbooking outcome per voyage.

4. overbooking_simulation.csv
   Overbooked voyages and estimated meters/bookings to move.

5. smart_recommendations.csv
   Recommended bookings/customers to defer based on smart deferral score.

6. feature_importance.csv
   Feature importance from the NoShow prediction model.

7. kpi_summary.csv
   Main KPI values for dashboard cards.

Suggested Power BI Dashboard Pages:
- Executive Summary
- Customer Risk Analysis
- Voyage and Route Analysis
- Overbooking Simulation
- Smart Recommendations
- Machine Learning Insights
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_text)

print("\nREADME created:")
print(readme_path)

print("\nPower BI export completed.")
print("Files saved in:")
print(POWERBI_DIR)