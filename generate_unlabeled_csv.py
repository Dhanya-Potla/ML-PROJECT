import csv
import random

# Output file
OUT_PATH = "sample_engine_fault_unlabeled.csv"

# Columns (same as dataset minus target)
COLUMNS = [
    "Vibration_Amplitude",
    "RMS_Vibration",
    "Vibration_Frequency",
    "Surface_Temperature",
    "Exhaust_Temperature",
    "Acoustic_dB",
    "Acoustic_Frequency",
    "Intake_Pressure",
    "Exhaust_Pressure",
    "Frequency_Band_Energy",
    "Amplitude_Mean",
]

# Value ranges approximated from your dataset summary
def make_row():
    return [
        round(random.uniform(0.1, 10.0), 3),          # Vibration_Amplitude
        round(random.uniform(0.05, 5.0), 3),          # RMS_Vibration
        round(random.uniform(20.0, 2000.0), 3),       # Vibration_Frequency
        round(random.uniform(30.0, 150.0), 3),        # Surface_Temperature
        round(random.uniform(200.0, 600.0), 3),       # Exhaust_Temperature
        round(random.uniform(60.0, 120.0), 3),        # Acoustic_dB
        round(random.uniform(100.0, 5000.0), 3),      # Acoustic_Frequency
        round(random.uniform(90.0, 120.0), 3),        # Intake_Pressure
        round(random.uniform(80.0, 110.0), 3),        # Exhaust_Pressure
        round(random.uniform(0.10, 1.00), 3),         # Frequency_Band_Energy
        round(random.uniform(0.01, 0.50), 3),         # Amplitude_Mean
    ]

def main(n_rows: int = 300):
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for _ in range(n_rows):
            writer.writerow(make_row())
    print(f"Wrote {n_rows} rows to {OUT_PATH}")

if __name__ == "__main__":
    main(300)


