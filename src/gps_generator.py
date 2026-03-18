"""
gps_generator.py
Generates synthetic GPS probe data from a running SUMO simulation via TraCI.
Outputs a pandas DataFrame with columns:
    time, vehicle_id, x, y, speed, vehicle_type, junction_proximity

The data is intentionally sparse (1 sample / 5 s per vehicle) and noisy
to mimic real-world GPS phone probes.
"""

import random
import numpy as np
import pandas as pd
import traci

# Junction ground-truth positions (metres, SUMO coordinate system)
JUNCTION_POSITIONS = {
    "J0": (0.0,   0.0),
    "J1": (400.0, 0.0),
    "J2": (800.0, 0.0),
}

GPS_NOISE_STD = 2.0        # metres of Gaussian noise
SAMPLE_INTERVAL = 5        # seconds between samples per vehicle
PROBE_FRACTION  = 0.6      # only 60 % of vehicles are "probe" vehicles


def _add_noise(x: float, y: float) -> tuple[float, float]:
    """Add Gaussian GPS noise."""
    return (
        x + random.gauss(0, GPS_NOISE_STD),
        y + random.gauss(0, GPS_NOISE_STD),
    )


def _closest_junction(x: float, y: float) -> str:
    """Return the id of the nearest junction."""
    return min(
        JUNCTION_POSITIONS,
        key=lambda j: (JUNCTION_POSITIONS[j][0] - x) ** 2
                    + (JUNCTION_POSITIONS[j][1] - y) ** 2,
    )


def collect_gps_frame(step: int, probe_vehicle_ids: set) -> list[dict]:
    """
    Called every SAMPLE_INTERVAL steps while TraCI is active.
    Returns a list of GPS records for vehicles in *probe_vehicle_ids*.
    """
    if step % SAMPLE_INTERVAL != 0:
        return []

    records = []
    for vid in traci.vehicle.getIDList():
        if vid not in probe_vehicle_ids:
            continue
        x, y = traci.vehicle.getPosition(vid)
        nx, ny = _add_noise(x, y)
        records.append(
            {
                "time":              step,
                "vehicle_id":        vid,
                "x":                 round(nx, 2),
                "y":                 round(ny, 2),
                "speed":             round(traci.vehicle.getSpeed(vid), 2),
                "vehicle_type":      traci.vehicle.getTypeID(vid),
                "junction_proximity": _closest_junction(x, y),
            }
        )
    return records


def select_probe_vehicles(all_ids: list[str]) -> set:
    """Randomly designate PROBE_FRACTION of vehicles as probes."""
    n = max(1, int(len(all_ids) * PROBE_FRACTION))
    return set(random.sample(all_ids, min(n, len(all_ids))))


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert raw records list to a clean DataFrame."""
    if not records:
        return pd.DataFrame(
            columns=["time", "vehicle_id", "x", "y",
                     "speed", "vehicle_type", "junction_proximity"]
        )
    return pd.DataFrame(records)
