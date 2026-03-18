"""
heatmap.py
Generates neon-styled Folium heatmaps from GPS probe DataFrames.

Coordinate mapping:
  SUMO x ∈ [-200, 1000]  →  longitude  ∈ [77.10, 77.12] (Bangalore-ish)
  SUMO y ∈ [-200,  200]  →  latitude   ∈ [12.97, 12.98]
"""

import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np

# ── Coordinate mapping ────────────────────────────────────────────────────────
# Map SUMO metres to approximate lat/lon (Bangalore area)
SUMO_X_MIN, SUMO_X_MAX = -200.0, 1000.0
SUMO_Y_MIN, SUMO_Y_MAX = -200.0,  200.0
LAT_MIN, LAT_MAX = 12.970, 12.980
LON_MIN, LON_MAX = 77.100, 77.120

# Junction lat/lon (for markers)
JUNCTION_LATLON = {
    "J0": (12.975, 77.1040),
    "J1": (12.975, 77.1080),
    "J2": (12.975, 77.1120),
}

HEAT_GRADIENT = {
    0.0:  "#0A0F1E",   # dark background
    0.25: "#00E5FF",   # cyan
    0.5:  "#7C4DFF",   # violet
    0.75: "#FF2FD6",   # magenta
    1.0:  "#FF0000",   # red (hotspot)
}


def _sumo_to_latlon(x: float, y: float) -> tuple[float, float]:
    lat = LAT_MIN + (y - SUMO_Y_MIN) / (SUMO_Y_MAX - SUMO_Y_MIN) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (x - SUMO_X_MIN) / (SUMO_X_MAX - SUMO_X_MIN) * (LON_MAX - LON_MIN)
    return lat, lon


def _weight(speed: float, max_speed: float = 14.0) -> float:
    """Heatmap intensity: slow vehicles = high weight (congestion)."""
    return max(0.1, 1.0 - speed / max_speed)


def build_heatmap(
    gps_df: pd.DataFrame,
    title: str = "Traffic Heatmap",
    show_junctions: bool = True,
) -> folium.Map:
    """
    Build and return a styled Folium heatmap from GPS probe data.

    Parameters
    ----------
    gps_df  : DataFrame with columns x, y, speed
    title   : map title (shown as tile layer name)
    show_junctions : whether to add junction markers
    """
    center_lat = (LAT_MIN + LAT_MAX) / 2
    center_lon = (LON_MIN + LON_MAX) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    if gps_df.empty:
        return m

    # Build heat data: [[lat, lon, weight], ...]
    heat_data = []
    for _, row in gps_df.iterrows():
        lat, lon = _sumo_to_latlon(row["x"], row["y"])
        w = _weight(row.get("speed", 5.0))
        heat_data.append([lat, lon, w])

    HeatMap(
        heat_data,
        name=title,
        min_opacity=0.3,
        max_zoom=18,
        radius=18,
        blur=15,
        gradient=HEAT_GRADIENT,
    ).add_to(m)

    if show_junctions:
        for jid, (jlat, jlon) in JUNCTION_LATLON.items():
            folium.CircleMarker(
                location=[jlat, jlon],
                radius=10,
                color="#00E5FF",
                fill=True,
                fill_color="#00E5FF",
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b style='color:#00E5FF'>{jid}</b>",
                    max_width=120,
                ),
                tooltip=jid,
            ).add_to(m)

    # Corridor road line
    road_coords = [
        _sumo_to_latlon(-200, 0),
        _sumo_to_latlon(1000, 0),
    ]
    folium.PolyLine(
        road_coords,
        color="#00E5FF",
        weight=3,
        opacity=0.5,
        tooltip="Main arterial",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def heatmap_to_html(
    gps_df: pd.DataFrame,
    title: str = "Traffic Heatmap",
) -> str:
    """Return heatmap as an HTML string for embedding in Streamlit."""
    m = build_heatmap(gps_df, title)
    return m._repr_html_()


def per_junction_density(gps_df: pd.DataFrame) -> dict[str, float]:
    """
    Returns average congestion score (1 - normalised speed) per junction.
    Used for dashboard KPI cards.
    """
    if gps_df.empty:
        return {"J0": 0.0, "J1": 0.0, "J2": 0.0}
    result = {}
    for jid in ["J0", "J1", "J2"]:
        subset = gps_df[gps_df["junction_proximity"] == jid]
        if subset.empty:
            result[jid] = 0.0
        else:
            avg_speed = subset["speed"].mean()
            result[jid] = round(max(0, 1 - avg_speed / 14.0), 3)
    return result
