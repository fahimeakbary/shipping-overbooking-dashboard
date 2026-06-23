from pathlib import Path

import pandas as pd
import streamlit as st
import joblib


# ==============================
# Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = BASE_DIR / "dashboard"
FEATURE_DIR = BASE_DIR / "data" / "features"
MODEL_PATH = BASE_DIR / "models" / "temporal_noshow_random_forest_model.pkl"


# ==============================
# Page configuration
# ==============================

st.set_page_config(
    page_title="Shipping Overbooking Dashboard",
    layout="wide"
)

st.title("Shipping Overbooking & NoShow Analytics Dashboard")


# ==============================
# Load data
# ==============================

kpi = pd.read_csv(DASHBOARD_DIR / "kpi_summary.csv")
risky_customers = pd.read_csv(DASHBOARD_DIR / "top_risky_customers.csv")
overbooked_voyages = pd.read_csv(DASHBOARD_DIR / "top_overbooked_voyages.csv")
recommendations = pd.read_csv(DASHBOARD_DIR / "top_smart_recommendations.csv")
customer_features = pd.read_csv(FEATURE_DIR / "customer_features_clean.csv")


# ==============================
# Load model
# ==============================

model = None

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


# ==============================
# Helper functions
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


def prepare_raw_booking_features(raw_data, customer_history):
    """
    Convert raw booking data into model-ready features.
    The uploaded file may include Booking Status, but it is not required.
    """

    history_cols = [
        "Customer Number",
        "Total_Final_Bookings",
        "NoShow_Count",
        "Loaded_Bookings",
        "NoShow_Rate",
        "Booking_Reliability_Score",
        "Reserved_Meter",
        "Loaded_Meter"
    ]

    enriched = raw_data.merge(
        customer_history[history_cols],
        on="Customer Number",
        how="left",
        suffixes=("", "_history")
    )

    # Fill missing history for new customers
    enriched["Total_Final_Bookings"] = enriched["Total_Final_Bookings"].fillna(0)
    enriched["NoShow_Count"] = enriched["NoShow_Count"].fillna(0)
    enriched["Loaded_Bookings"] = enriched["Loaded_Bookings"].fillna(0)
    enriched["NoShow_Rate"] = enriched["NoShow_Rate"].fillna(0)
    enriched["Booking_Reliability_Score"] = enriched["Booking_Reliability_Score"].fillna(0)

    # Because uploaded data already has Reserved Meter, history columns may get suffixes
    if "Reserved_Meter_history" in enriched.columns:
        enriched["Reserved_Meter_history"] = enriched["Reserved_Meter_history"].fillna(0)
        historical_reserved_meter = enriched["Reserved_Meter_history"]
    else:
        historical_reserved_meter = 0

    if "Loaded_Meter" in enriched.columns:
        enriched["Loaded_Meter"] = enriched["Loaded_Meter"].fillna(0)
        historical_loaded_meter = enriched["Loaded_Meter"]
    else:
        historical_loaded_meter = 0

    # Create temporal features required by the model
    enriched["Historical_Bookings"] = enriched["Total_Final_Bookings"]
    enriched["Historical_NoShows"] = enriched["NoShow_Count"]
    enriched["Historical_Loaded"] = enriched["Loaded_Bookings"]
    enriched["Historical_NoShow_Rate"] = enriched["NoShow_Rate"]
    enriched["Historical_Reliability"] = enriched["Booking_Reliability_Score"]
    enriched["Historical_Reserved_Meter"] = historical_reserved_meter
    enriched["Historical_Loaded_Meter"] = historical_loaded_meter

    if "Days_Since_Last_Booking" not in enriched.columns:
        enriched["Days_Since_Last_Booking"] = 0

    return enriched


def run_prediction(input_data, required_columns):
    """
    Run NoShow prediction.
    """

    X_new = input_data[required_columns].copy()

    input_data["NoShow_Probability"] = model.predict_proba(X_new)[:, 1]
    input_data["Predicted_NoShow"] = model.predict(X_new)

    input_data["Prediction_Risk_Category"] = input_data[
        "NoShow_Probability"
    ].apply(assign_prediction_risk)

    return input_data


def show_prediction_summary(predicted_data):
    """
    Show prediction results as KPI cards and risk interpretation.
    """

    avg_probability = predicted_data["NoShow_Probability"].mean()

    predicted_noshow_count = int(
        predicted_data["Predicted_NoShow"].sum()
    )

    high_risk_count = int(
        (
            predicted_data["Prediction_Risk_Category"]
            == "High NoShow Risk"
        ).sum()
    )

    medium_risk_count = int(
        (
            predicted_data["Prediction_Risk_Category"]
            == "Medium NoShow Risk"
        ).sum()
    )

    low_risk_count = int(
        (
            predicted_data["Prediction_Risk_Category"]
            == "Low NoShow Risk"
        ).sum()
    )

    st.subheader("Prediction Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    summary_col1.metric(
        "Average NoShow Probability",
        f"{avg_probability:.2%}"
    )

    summary_col2.metric(
        "Predicted NoShows",
        predicted_noshow_count
    )

    summary_col3.metric(
        "High Risk",
        high_risk_count
    )

    summary_col4.metric(
        "Low Risk",
        low_risk_count
    )

    if high_risk_count > 0:
        st.error(
            f"{high_risk_count} booking(s) are classified as High NoShow Risk. "
            "These bookings should be reviewed before final capacity planning."
        )
    elif medium_risk_count > 0:
        st.warning(
            f"{medium_risk_count} booking(s) are classified as Medium NoShow Risk. "
            "These bookings may need monitoring."
        )
    else:
        st.success(
            "All uploaded bookings are classified as Low NoShow Risk."
        )

    high_risk_df = predicted_data[
        predicted_data["Prediction_Risk_Category"] == "High NoShow Risk"
    ]

    if not high_risk_df.empty:
        st.subheader("High Risk Bookings")

        high_risk_display_cols = [
            "Customer Number",
            "Departure Port",
            "Arrival Port",
            "Sail Date",
            "Sail Time",
            "Ship Code",
            "Reserved Meter",
            "NoShow_Probability",
            "Prediction_Risk_Category"
        ]

        high_risk_display_cols = [
            col for col in high_risk_display_cols
            if col in high_risk_df.columns
        ]

        st.dataframe(
            high_risk_df[high_risk_display_cols],
            width="stretch"
        )


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
# Tabs
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
# Tab 1
# ==============================

with tab1:
    st.header("Top Risky Customers")
    st.dataframe(risky_customers, width="stretch")


# ==============================
# Tab 2
# ==============================

with tab2:
    st.header("Top Overbooked Voyages")
    st.dataframe(overbooked_voyages, width="stretch")


# ==============================
# Tab 3
# ==============================

with tab3:
    st.header("Smart Overbooking Recommendations")
    st.dataframe(recommendations, width="stretch")


# ==============================
# Tab 4
# ==============================

with tab4:
    st.header("Predict NoShow Risk for New Booking Data")

    if model is None:
        st.warning(
            "The trained ML model file is not available. "
            "Live prediction is disabled."
        )
    else:
        input_mode = st.radio(
            "Choose input method:",
            ["Upload Booking CSV", "Manual Input"]
        )

        upload_required_columns = [
            "Customer Number",
            "Departure Port",
            "Arrival Port",
            "Sail Date",
            "Sail Time",
            "Ship Code",
            "Reserved Meter",
            "Reserved Heads"
        ]

        model_required_columns = [
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
            "Booking Status",
            "Reserved Meter",
            "Reserved Heads",
            "Historical_Bookings",
            "Historical_NoShow_Rate",
            "Historical_Reliability",
            "NoShow_Probability",
            "Predicted_NoShow",
            "Prediction_Risk_Category"
        ]

        # ==============================
        # Option A: Upload booking CSV
        # ==============================

        if input_mode == "Upload Booking CSV":

            st.subheader("Upload Booking CSV")

            st.write(
                "The uploaded file should contain booking-level columns. "
                "Booking Status is optional."
            )

            st.code(
                "Customer Number,Departure Port,Arrival Port,Sail Date,Sail Time,Ship Code,Reserved Meter,Reserved Heads,Booking Status\n"
                "C00542,Rostock,Trelleborg,2026-06-10,18:00,SHIP_03,14.5,1,Booked",
                language="csv"
            )

            uploaded_file = st.file_uploader(
                "Upload booking file",
                type=["csv"],
                key="booking_upload"
            )

            if uploaded_file is not None:
                raw_data = pd.read_csv(uploaded_file)

                missing_columns = [
                    col for col in upload_required_columns
                    if col not in raw_data.columns
                ]

                if missing_columns:
                    st.error("Missing required columns:")
                    st.write(missing_columns)
                else:
                    prepared_data = prepare_raw_booking_features(
                        raw_data,
                        customer_features
                    )

                    predicted_data = run_prediction(
                        prepared_data,
                        model_required_columns
                    )

                    st.success("Prediction completed!")

                    show_prediction_summary(predicted_data)

                    existing_display_columns = [
                        col for col in display_columns
                        if col in predicted_data.columns
                    ]

                    st.subheader("Full Prediction Results")

                    st.dataframe(
                        predicted_data[existing_display_columns],
                        width="stretch"
                    )

                    csv = predicted_data.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label="Download Prediction Results as CSV",
                        data=csv,
                        file_name="booking_predictions.csv",
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

            if historical_bookings > 0:
                historical_noshow_rate = historical_noshows / historical_bookings
            else:
                historical_noshow_rate = 0

            if historical_reserved_meter > 0:
                historical_reliability = historical_loaded_meter / historical_reserved_meter
            else:
                historical_reliability = 0

            historical_noshow_rate = min(max(historical_noshow_rate, 0), 1)
            historical_reliability = min(max(historical_reliability, 0), 1)

            st.info(
                f"Historical NoShow Rate: {historical_noshow_rate:.2%} | "
                f"Meter-based Reliability: {historical_reliability:.2%}"
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

                predicted_data = run_prediction(
                    manual_data,
                    model_required_columns
                )

                show_prediction_summary(predicted_data)

                existing_display_columns = [
                    col for col in display_columns
                    if col in predicted_data.columns
                ]

                st.subheader("Full Prediction Results")

                st.dataframe(
                    predicted_data[existing_display_columns],
                    width="stretch"
                )

st.divider()

st.subheader("Project Summary")

st.write(
    """
    This dashboard supports shipping capacity planning by combining customer reliability analysis,
    voyage-level capacity utilization, overbooking simulation, machine learning-based NoShow prediction,
    and smart recommendation logic for deferring bookings.
    """
)