"""
Feature engineering module for Freight Rate Prediction.
Calculates geospatial, transit, temporal, cargo, and market interaction features.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate the great circle distance in miles between points on Earth."""
    r = 3958.8  # Earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def engineer_features(
    df: pd.DataFrame,
    lane_stats: dict[str, float] | None = None,
    equipment_quote_medians: dict[str, float] | None = None,
    is_training: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Constructs a rich tabular feature set for tree-based models.
    """
    df = df.copy()

    # Geospatial features
    lat1 = df["pickup_lat"].values
    lon1 = df["pickup_lon"].values
    lat2 = df["delivery_lat"].values
    lon2 = df["delivery_lon"].values

    hav = haversine_np(lat1, lon1, lat2, lon2)
    df["haversine_dist"] = hav
    df["circuitousness"] = df["distance"] / (hav + 1.0)
    df["delta_lat"] = lat2 - lat1
    df["delta_lon"] = lon2 - lon1
    df["transit_bearing"] = np.arctan2(df["delta_lon"], df["delta_lat"])

    # Load & Cargo features
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1.0)

    # Temporal features
    dayofweek = df["dayofweek"].values
    dayofyear = df["dayofyear"].values
    df["sin_dayofweek"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * dayofweek / 7.0)
    df["sin_dayofyear"] = np.sin(2 * np.pi * dayofyear / 365.25)
    df["cos_dayofyear"] = np.cos(2 * np.pi * dayofyear / 365.25)

    # Q4 / Holiday indicators
    # Thanksgiving week (~late Nov), Christmas (Dec 24-26), New Year (Dec 31)
    month = df["month"].values
    day = df["day"].values
    df["is_christmas_period"] = ((month == 12) & (day >= 23) & (day <= 26)).astype(int)
    df["is_new_year_eve"] = ((month == 12) & (day >= 30)).astype(int)
    df["is_thanksgiving_period"] = ((month == 11) & (day >= 24) & (day <= 28)).astype(int)

    # Lane identification
    df["lane"] = df["pickup"].astype(str) + " -> " + df["delivery"].astype(str)
    df["lane_equipment"] = df["lane"] + " (" + df["equipment"].astype(str) + ")"

    # Quote signal handling: if missing, use lane-equipment historical statistics
    if is_training:
        lane_stats = df.groupby("lane_equipment")["quote_signal"].median().to_dict()
        equipment_quote_medians = df.groupby("equipment")["quote_signal"].median().to_dict()

    if "quote_signal" not in df or df["quote_signal"].isna().any():
        if "quote_signal" not in df:
            df["quote_signal"] = np.nan
        lane_fallback = df["lane_equipment"].map(lane_stats or {})
        equip_fallback = df["equipment"].map(equipment_quote_medians or {})
        df["quote_signal"] = df["quote_signal"].fillna(lane_fallback).fillna(equip_fallback).fillna(2.05)

    # Market & Quote signal interactions
    df["base_quote"] = df["distance"] * df["quote_signal"]
    df["market_interaction"] = df["quote_signal"] * df["market_index"]
    df["distance_market"] = df["distance"] * df["market_index"]

    feature_metadata = {
        "lane_stats": lane_stats,
        "equipment_quote_medians": equipment_quote_medians,
    }
    return df, feature_metadata


FEATURE_COLUMNS = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "base_quote",
    "haversine_dist",
    "circuitousness",
    "delta_lat",
    "delta_lon",
    "transit_bearing",
    "weight_per_mile",
    "market_interaction",
    "distance_market",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "weekofyear",
    "is_weekend",
    "is_month_end",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
    "is_christmas_period",
    "is_new_year_eve",
    "is_thanksgiving_period",
]

CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
]
