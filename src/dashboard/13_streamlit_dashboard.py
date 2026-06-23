import os
import pandas as pd
import streamlit as st
import joblib

# ==============================
# Paths
# ==============================

BASE_DIR = r"C:\laptop\shipping_booking_project"
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "temporal_noshow_random_forest_model.pkl"
)

# ==============================
# Page configuration
# ==============================

st.set_page_config(
    page_title="Shipping Overbooking Dashboard",
    layout="wide"
)

st.title("Shipping Overbooking & NoShow Analytics Dashboard")

# ==============================
# Load dashboard data
# ==============================

kpi = pd.read_csv(os.path.join(DASHBOARD_DIR, "kpi_summary.csv"))
risky_customers = pd.read_csv(os.path.join(DASHBOARD_DIR, "top_risky_customers.csv"))
overbooked_voyages = pd.read_csv(os.path.join(DASHBOARD_DIR, "top_overbooked_voyages.csv"))
recommendations = pd.read_csv(os.path.join(DASHBOARD_DIR, "top_smart_recommendations.csv"))

# ==============================
# Load trained model
# ==============================

model = joblib.load(MODEL_PATH)

# ==============================
# Helper function
# ==============================

def get_value(metric_name):
    return kpi.loc[kpi["Metric"] == metric_name, "Value"].values[0]


def assign_prediction_risk(prob):
    if prob >= 0.70:
        return "High NoShow Risk"
    elif prob >= 0.40:
        return "Medium NoShow Risk"
    else:
        return "Low NoShow Risk"


# ==============================
# KPI section
# ==============================

st.header("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Voyages", int(get_value("Total Voyages")))
col2.metric("Overbooked Voyages", int(get_value("Overbooked Voyages")))
col3.metric("High Risk Customers", int(get_value("High Risk Customers")))
col4.metric("Total Recommendations", int(get_value("Total Recommendations")))

col5, col6, col7 = st.columns(3)

col5.metric(
    "Avg Capacity Utilization",
    f"{get_value('Average Capacity Utilization'):.2%}"
)

col6.metric(
    "Avg Wasted Capacity",
    f"{get_value('Average Wasted Capacity'):.2%}"
)

col7.metric(
    "Avg NoShow Probability",
    f"{get_value('Average NoShow Probability'):.2%}"
)

st.divider()

# ==============================
# Dashboard tabs
# ==============================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Risky Customers",
        "Overbooked Voyages",
        "Smart Recommendations",
        "New Booking Prediction"
    ]
)

# ==============================
# Tab 1: Risky customers
# ==============================

with tab1:
    st.header("Top Risky Customers")
    st.write(
        "Customers ranked by Deferral Candidate Score. Higher score means a better candidate for deferral."
    )
    st.dataframe(risky_customers, width="stretch")

# ==============================
# Tab 2: Overbooked voyages
# ==============================

with tab2:
    st.header("Top Overbooked Voyages")
    st.write("Voyages ranked by overbooked meters.")
    st.dataframe(overbooked_voyages, width="stretch")

# ==============================
# Tab 3: Smart recommendations
# ==============================

with tab3:
    st.header("Smart Overbooking Recommendations")
    st.write(
        "Recommended bookings to defer based on customer risk and ML-based NoShow probability."
    )
    st.dataframe(recommendations, width="stretch")

# ==============================
# Tab 4: New booking prediction
# ==============================

with tab4:
    st.header("Predict NoShow Risk for New Booking Data")

    input_mode = st.radio(
        "Choose input method:",
        ["Upload CSV", "Manual Input"]
    )

    required_columns = [
        "Departure Port",
        "Arrival Port",
        "Sail Time",
        "Ship Code",
        "Reserved Meter",
        "Reserved Heads",
        "Historical_Bookings",
        "Historical_NoShows",
        "Historical_Loaded",
        "Historical_NoShow_Rate",
        "Historical_Reliability",
        "Historical_Reserved_Meter",
        "Historical_Loaded_Meter",
        "Days_Since_Last_Booking"
    ]

    display_columns = [
        "Customer Number",
        "Departure Port",
        "Arrival Port",
        "Sail Date",
        "Sail Time",
        "Ship Code",
        "Reserved Meter",
        "Reserved Heads",
        "NoShow_Probability",
        "Predicted_NoShow",
        "Prediction_Risk_Category"
    ]

    # ==============================
    # Option A: Upload CSV
    # ==============================

    if input_mode == "Upload CSV":

        st.subheader("Upload New Booking CSV")

        st.write(
            "Upload a CSV file with new bookings. The file must contain the required model feature columns."
        )

        uploaded_file = st.file_uploader(
            "Upload new_bookings.csv",
            type=["csv"]
        )

        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)

            missing_columns = [
                col for col in required_columns
                if col not in new_data.columns
            ]

            if missing_columns:
                st.error("Missing required columns:")
                st.write(missing_columns)
            else:
                X_new = new_data[required_columns]

                new_data["NoShow_Probability"] = model.predict_proba(X_new)[:, 1]
                new_data["Predicted_NoShow"] = model.predict(X_new)

                new_data["Prediction_Risk_Category"] = new_data[
                    "NoShow_Probability"
                ].apply(assign_prediction_risk)

                st.success("Prediction completed!")

                existing_display_columns = [
                    col for col in display_columns
                    if col in new_data.columns
                ]

                st.dataframe(
                    new_data[existing_display_columns],
                    width="stretch"
                )

                csv = new_data.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    label="Download Prediction Results as CSV",
                    data=csv,
                    file_name="new_booking_predictions.csv",
                    mime="text/csv"
                )

    # ==============================
    # Option B: Manual input
    # ==============================

    elif input_mode == "Manual Input":

        st.subheader("Manual Booking Input")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            customer_number = st.text_input("Customer Number", "C00542")
            departure_port = st.selectbox(
                "Departure Port",
                ["Travemünde", "Trelleborg", "Rostock", "Klaipeda", "Swinoujscie"]
            )
            arrival_port = st.selectbox(
                "Arrival Port",
                ["Travemünde", "Trelleborg", "Rostock", "Klaipeda", "Swinoujscie"]
            )
            sail_date = st.date_input("Sail Date")

        with col_b:
            sail_time = st.selectbox(
                "Sail Time",
                ["06:00", "08:00", "10:00", "14:00", "18:00", "22:00"]
            )
            ship_code = st.selectbox(
                "Ship Code",
                ["SHIP_01", "SHIP_02", "SHIP_03", "SHIP_04", "SHIP_05"]
            )
            reserved_meter = st.number_input(
                "Reserved Meter",
                min_value=0.0,
                value=14.5,
                step=0.5
            )
            reserved_heads = st.number_input(
                "Reserved Heads",
                min_value=0,
                value=1,
                step=1
            )

        with col_c:
            historical_bookings = st.number_input(
                "Historical Bookings",
                min_value=0,
                value=49,
                step=1
            )
            historical_noshows = st.number_input(
                "Historical NoShows",
                min_value=0,
                value=34,
                step=1
            )
            historical_loaded = st.number_input(
                "Historical Loaded",
                min_value=0,
                value=15,
                step=1
            )
            days_since_last_booking = st.number_input(
                "Days Since Last Booking",
                min_value=0,
                value=12,
                step=1
            )

        historical_reserved_meter = st.number_input(
            "Historical Reserved Meter",
            min_value=0.0,
            value=707.9,
            step=1.0
        )

        historical_loaded_meter = st.number_input(
            "Historical Loaded Meter",
            min_value=0.0,
            value=80.4,
            step=1.0
        )

        # ==============================
        # Calculate historical rates
        # ==============================

        if historical_bookings > 0:
            historical_noshow_rate = historical_noshows / historical_bookings
            historical_reliability = historical_loaded / historical_bookings
        else:
            historical_noshow_rate = 0
            historical_reliability = 0

        st.info(
            f"Historical NoShow Rate: {historical_noshow_rate:.2%} | "
            f"Historical Reliability: {historical_reliability:.2%}"
        )

        if st.button("Predict NoShow Risk"):

            manual_data = pd.DataFrame(
                {
                    "Customer Number": [customer_number],
                    "Departure Port": [departure_port],
                    "Arrival Port": [arrival_port],
                    "Sail Date": [str(sail_date)],
                    "Sail Time": [sail_time],
                    "Ship Code": [ship_code],
                    "Reserved Meter": [reserved_meter],
                    "Reserved Heads": [reserved_heads],
                    "Historical_Bookings": [historical_bookings],
                    "Historical_NoShows": [historical_noshows],
                    "Historical_Loaded": [historical_loaded],
                    "Historical_NoShow_Rate": [historical_noshow_rate],
                    "Historical_Reliability": [historical_reliability],
                    "Historical_Reserved_Meter": [historical_reserved_meter],
                    "Historical_Loaded_Meter": [historical_loaded_meter],
                    "Days_Since_Last_Booking": [days_since_last_booking]
                }
            )

            X_manual = manual_data[required_columns]

            manual_data["NoShow_Probability"] = model.predict_proba(X_manual)[:, 1]
            manual_data["Predicted_NoShow"] = model.predict(X_manual)

            manual_data["Prediction_Risk_Category"] = manual_data[
                "NoShow_Probability"
            ].apply(assign_prediction_risk)

            st.success("Prediction completed!")

            probability = manual_data.loc[0, "NoShow_Probability"]
            risk_category = manual_data.loc[0, "Prediction_Risk_Category"]
            predicted_noshow = manual_data.loc[0, "Predicted_NoShow"]

            col_result1, col_result2, col_result3 = st.columns(3)

            col_result1.metric(
                "NoShow Probability",
                f"{probability:.2%}"
            )

            col_result2.metric(
                "Predicted NoShow",
                int(predicted_noshow)
            )

            col_result3.metric(
                "Risk Category",
                risk_category
            )

            st.dataframe(
                manual_data[display_columns],
                width="stretch"
            )

            csv = manual_data.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download Manual Prediction as CSV",
                data=csv,
                file_name="manual_booking_prediction.csv",
                mime="text/csv"
            )

st.divider()

# ==============================
# Project summary
# ==============================

st.subheader("Project Summary")

st.write(
    """
    This dashboard supports shipping capacity planning by combining customer reliability analysis,
    voyage-level capacity utilization, overbooking simulation, machine learning-based NoShow prediction,
    and smart recommendation logic for deferring bookings.
    """
)