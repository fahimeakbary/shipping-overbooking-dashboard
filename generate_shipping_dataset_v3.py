
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(2026)

BASE_DIR = Path(r"C:\laptop\shipping_booking_project")
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

n_bookings = 60000
n_customers = 1200

routes = [
    ("Travemünde", "Trelleborg"),
    ("Trelleborg", "Travemünde"),
    ("Rostock", "Trelleborg"),
    ("Trelleborg", "Rostock"),
    ("Travemünde", "Klaipeda"),
    ("Klaipeda", "Travemünde"),
    ("Rostock", "Klaipeda"),
    ("Klaipeda", "Rostock"),
    ("Trelleborg", "Swinoujscie"),
    ("Swinoujscie", "Trelleborg"),
]

ships = [
    {"Ship Code": "SHIP_01", "Total Meter": 1800, "Total Head": 180},
    {"Ship Code": "SHIP_02", "Total Meter": 2000, "Total Head": 220},
    {"Ship Code": "SHIP_03", "Total Meter": 2200, "Total Head": 250},
    {"Ship Code": "SHIP_04", "Total Meter": 2400, "Total Head": 280},
    {"Ship Code": "SHIP_05", "Total Meter": 2600, "Total Head": 300},
    {"Ship Code": "SHIP_06", "Total Meter": 2100, "Total Head": 230},
    {"Ship Code": "SHIP_07", "Total Meter": 1900, "Total Head": 200},
    {"Ship Code": "SHIP_08", "Total Meter": 2300, "Total Head": 260},
]

ship_capacity = pd.DataFrame(ships)
customers = [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)]

customer_reliability = {}
for c in customers:
    group = np.random.choice(["reliable", "average", "unreliable"], p=[0.35, 0.45, 0.20])
    if group == "reliable":
        customer_reliability[c] = np.random.uniform(0.88, 0.98)
    elif group == "average":
        customer_reliability[c] = np.random.uniform(0.68, 0.88)
    else:
        customer_reliability[c] = np.random.uniform(0.38, 0.68)

n_sailings = 2500
sail_dates = pd.to_datetime(
    np.random.choice(pd.date_range("2024-01-01", "2025-12-31", freq="D"), n_sailings)
)
sail_times = np.random.choice(["06:00", "08:00", "10:00", "14:00", "18:00", "22:00"], n_sailings)
route_idx = np.random.choice(range(len(routes)), n_sailings)
ship_codes = np.random.choice(ship_capacity["Ship Code"], n_sailings)

sailings = pd.DataFrame({
    "Sailing ID": [f"S{str(i).zfill(6)}" for i in range(1, n_sailings + 1)],
    "Departure Port": [routes[i][0] for i in route_idx],
    "Arrival Port": [routes[i][1] for i in route_idx],
    "Sail Date": sail_dates.date.astype(str),
    "Sail Time": sail_times,
    "Ship Code": ship_codes,
}).merge(ship_capacity, on="Ship Code", how="left")

selected_sailing_idx = np.random.choice(sailings.index, n_bookings)
selected_sailings = sailings.loc[selected_sailing_idx].reset_index(drop=True)

customer_numbers = np.random.choice(customers, n_bookings)

booking_statuses = []
reserved_meters = []
reserved_heads = []
loaded_meters = []
loaded_heads = []

for customer in customer_numbers:
    reliability = customer_reliability[customer]

    p_cancelled = 0.04
    p_noshow = max(0.02, 1 - reliability)
    p_checked_in = 0.08
    p_booked = 0.07
    p_loaded = max(0.0, 1 - (p_cancelled + p_noshow + p_checked_in + p_booked))

    status = np.random.choice(
        ["Loaded", "Checked In", "Booked", "NoShow", "Cancelled"],
        p=[p_loaded, p_checked_in, p_booked, p_noshow, p_cancelled]
    )

    rm = round(max(4.0, np.random.normal(14, 4)), 1)
    rh = np.random.choice([1, 2], p=[0.82, 0.18])

    if status == "Loaded":
        lm = round(rm * np.random.uniform(0.95, 1.00), 1)
        lh = rh
    elif status == "Checked In":
        lm = 0.0
        lh = rh
    else:
        lm = 0.0
        lh = 0

    booking_statuses.append(status)
    reserved_meters.append(rm)
    reserved_heads.append(rh)
    loaded_meters.append(lm)
    loaded_heads.append(lh)

bookings = pd.DataFrame({
    "Customer Number": customer_numbers,
    "Departure Port": selected_sailings["Departure Port"],
    "Arrival Port": selected_sailings["Arrival Port"],
    "Sail Date": selected_sailings["Sail Date"],
    "Sail Time": selected_sailings["Sail Time"],
    "Ship Code": selected_sailings["Ship Code"],
    "SeqNo": selected_sailings.groupby(["Sailing ID"]).cumcount() + 1,
    "Booking Status": booking_statuses,
    "Reserved Meter": reserved_meters,
    "Reserved Heads": reserved_heads,
    "Loaded Meter": loaded_meters,
    "Loaded Heads": loaded_heads,
    "Total Meter": selected_sailings["Total Meter"],
    "Total Head": selected_sailings["Total Head"],
})

bookings.to_csv(RAW_DIR / "shipping_bookings_v3_company_like.csv", index=False)
ship_capacity.to_csv(RAW_DIR / "ship_capacity_v3.csv", index=False)
sailings.to_csv(RAW_DIR / "sailings_v3.csv", index=False)

print("Files saved in:", RAW_DIR)
print(bookings.head())
