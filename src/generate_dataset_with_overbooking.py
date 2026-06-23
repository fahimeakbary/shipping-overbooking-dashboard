import os
import pandas as pd
import numpy as np

np.random.seed(2026)

BASE_DIR = r"C:\laptop\shipping_booking_project"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ==============================
# Basic settings
# ==============================

customers = [f"C{str(i).zfill(5)}" for i in range(1, 1201)]

ships = pd.DataFrame([
    {"Ship Code": "SHIP_01", "Total Meter": 1800, "Total Head": 180},
    {"Ship Code": "SHIP_02", "Total Meter": 2000, "Total Head": 220},
    {"Ship Code": "SHIP_03", "Total Meter": 2100, "Total Head": 230},
    {"Ship Code": "SHIP_04", "Total Meter": 2200, "Total Head": 250},
    {"Ship Code": "SHIP_05", "Total Meter": 2400, "Total Head": 280},
])

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

# hidden customer reliability only for simulation
customer_reliability = {
    c: np.random.choice(
        [np.random.uniform(0.85, 0.98),
         np.random.uniform(0.60, 0.85),
         np.random.uniform(0.35, 0.60)],
        p=[0.35, 0.45, 0.20]
    )
    for c in customers
}

records = []
n_sailings = 2500

for sailing_id in range(1, n_sailings + 1):

    departure, arrival = routes[np.random.randint(len(routes))]
    ship = ships.sample(1).iloc[0]

    ship_code = ship["Ship Code"]
    total_meter = ship["Total Meter"]
    total_head = ship["Total Head"]

    sail_date = np.random.choice(
        pd.date_range("2024-01-01", "2025-12-31", freq="D")
    )
    sail_time = np.random.choice(["06:00", "08:00", "10:00", "14:00", "18:00", "22:00"])

    # ==============================
    # Overbooking scenario
    # ==============================
    scenario = np.random.choice(
        ["normal", "mild_overbooked", "medium_overbooked", "high_overbooked"],
        p=[0.65, 0.18, 0.12, 0.05]
    )

    if scenario == "normal":
        target_reserved_meter = total_meter * np.random.uniform(0.75, 1.00)
    elif scenario == "mild_overbooked":
        target_reserved_meter = total_meter * np.random.uniform(1.01, 1.05)
    elif scenario == "medium_overbooked":
        target_reserved_meter = total_meter * np.random.uniform(1.06, 1.10)
    else:
        target_reserved_meter = total_meter * np.random.uniform(1.11, 1.20)

    current_reserved_meter = 0
    seq_no = 1

    while current_reserved_meter < target_reserved_meter:

        customer = np.random.choice(customers)
        reliability = customer_reliability[customer]

        reserved_meter = round(max(4.0, np.random.normal(14, 4)), 1)
        reserved_heads = int(np.random.choice([1, 2], p=[0.85, 0.15]))

        # status probabilities
        p_noshow = 1 - reliability
        p_cancelled = 0.03
        p_checked_in = 0.06
        p_booked = 0.04
        p_loaded = max(0.01, 1 - (p_noshow + p_cancelled + p_checked_in + p_booked))

        probs = np.array([p_loaded, p_noshow, p_checked_in, p_booked, p_cancelled])
        probs = probs / probs.sum()

        status = np.random.choice(
            ["Loaded", "NoShow", "Checked In", "Booked", "Cancelled"],
            p=probs
        )

        if status == "Loaded":
            loaded_meter = round(reserved_meter * np.random.uniform(0.95, 1.00), 1)
            loaded_heads = reserved_heads
        elif status == "Checked In":
            loaded_meter = 0.0
            loaded_heads = reserved_heads
        else:
            loaded_meter = 0.0
            loaded_heads = 0

        records.append({
            "Customer Number": customer,
            "Departure Port": departure,
            "Arrival Port": arrival,
            "Sail Date": pd.to_datetime(sail_date).date(),
            "Sail Time": sail_time,
            "Ship Code": ship_code,
            "SeqNo": seq_no,
            "Booking Status": status,
            "Reserved Meter": reserved_meter,
            "Reserved Heads": reserved_heads,
            "Loaded Meter": loaded_meter,
            "Loaded Heads": loaded_heads,
            "Total Meter": total_meter,
            "Total Head": total_head,
            "Overbooking Scenario": scenario
        })

        current_reserved_meter += reserved_meter
        seq_no += 1

bookings = pd.DataFrame(records)

output_path = os.path.join(
    RAW_DIR,
    "shipping_bookings_with_overbooking.csv"
)

bookings.to_csv(output_path, index=False)

print("=" * 70)
print("DATASET WITH OVERBOOKING CREATED")
print("=" * 70)

print("Shape:")
print(bookings.shape)

print("\nBooking Status Counts:")
print(bookings["Booking Status"].value_counts())

print("\nOverbooking Scenario Counts:")
print(bookings["Overbooking Scenario"].value_counts())

print("\nOutput saved to:")
print(output_path)