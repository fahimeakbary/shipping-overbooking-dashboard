import pandas as pd
import os


base_dir = r"C:\laptop\shipping_booking_project"

new_dir = os.path.join(base_dir, "data", "new")
os.makedirs(new_dir, exist_ok=True)

# داده نمونه
data = {
    "Customer Number": ["C00542", "C00869", "C00752"],
    "Departure Port": ["Rostock", "Travemünde", "Klaipeda"],
    "Arrival Port": ["Trelleborg", "Klaipeda", "Rostock"],
    "Sail Date": ["2026-06-10", "2026-06-11", "2026-06-12"],
    "Sail Time": ["18:00", "22:00", "08:00"],
    "Ship Code": ["SHIP_03", "SHIP_05", "SHIP_02"],
    "Reserved Meter": [14.5, 16.0, 13.0],
    "Reserved Heads": [1, 1, 2],
    "Historical_Bookings": [49, 49, 52],
    "Historical_NoShows": [34, 2, 36],
    "Historical_Loaded": [15, 47, 16],
    "Historical_NoShow_Rate": [0.6938, 0.0408, 0.6923],
    "Historical_Reliability": [0.3061, 0.9592, 0.3077],
    "Historical_Reserved_Meter": [707.9, 610.3, 763.2],
    "Historical_Loaded_Meter": [80.4, 549.4, 91.0],
    "Days_Since_Last_Booking": [12, 8, 5]
}

# DataFrame
df = pd.DataFrame(data)

output_path = os.path.join(new_dir, "new_bookings.csv")
df.to_csv(output_path, index=False)

print("File created successfully:")
print(output_path)

print("\nPreview:")
print(df)