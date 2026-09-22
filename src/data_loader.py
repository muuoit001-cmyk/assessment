"""
Data loading and preprocessing utilities for Freight Rate Prediction.
Handles data cleaning, sign-inversion fixes for weight, missing value imputation,
and coordinate extraction.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV path."""
    return pd.read_csv(path)


def clean_and_prepare(
    df: pd.DataFrame,
    city_coords: dict[str, tuple[float, float]] | None = None,
    equipment_weight_medians: dict[str, float] | None = None,
    daily_market_index_medians: dict[str, float] | None = None,
    global_market_median: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Cleans raw dataframe and applies systematic data quality fixes:
    1. Fix negative weights by taking absolute value (sign inversion error).
    2. Impute missing weights by equipment-type median.
    3. Impute missing market_index by date-level median.
    4. Ensure pickup/delivery coordinates are populated using reference mapping.
    5. Extract datetime attributes.
    """
    df = df.copy()

    # 1. Coordinate mapping dictionary construction / population
    if city_coords is None:
        city_coords = {}
        if "pickup" in df and "pickup_lat" in df and "pickup_lon" in df:
            valid_p = df.dropna(subset=["pickup", "pickup_lat", "pickup_lon"])
            for _, row in valid_p.drop_duplicates("pickup").iterrows():
                city_coords[row["pickup"]] = (float(row["pickup_lat"]), float(row["pickup_lon"]))
        if "delivery" in df and "delivery_lat" in df and "delivery_lon" in df:
            valid_d = df.dropna(subset=["delivery", "delivery_lat", "delivery_lon"])
            for _, row in valid_d.drop_duplicates("delivery").iterrows():
                if row["delivery"] not in city_coords:
                    city_coords[row["delivery"]] = (float(row["delivery_lat"]), float(row["delivery_lon"]))

    # Fill coordinates if missing (e.g., december_chart_inputs)
    if "pickup_lat" not in df or df["pickup_lat"].isna().any():
        df["pickup_lat"] = df["pickup"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
        df["pickup_lon"] = df["pickup"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])
    if "delivery_lat" not in df or df["delivery_lat"].isna().any():
        df["delivery_lat"] = df["delivery"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
        df["delivery_lon"] = df["delivery"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])

    # 2. Clean weights: fix negative weights (sign inversion)
    if "weight" in df:
        df["weight"] = df["weight"].abs()
        if equipment_weight_medians is None:
            equipment_weight_medians = df.groupby("equipment")["weight"].median().to_dict()
        df["weight"] = df["weight"].fillna(df["equipment"].map(equipment_weight_medians))
        # Global fallback if any equipment missing
        df["weight"] = df["weight"].fillna(31500.0)

    # 3. Clean market_index: date-level median imputation
    if "market_index" not in df:
        df["market_index"] = np.nan

    if daily_market_index_medians is None:
        daily_market_index_medians = df.groupby("date")["market_index"].median().dropna().to_dict()
        global_market_median = float(df["market_index"].median()) if not df["market_index"].isna().all() else 1.0

    # Impute missing by date
    date_impute = df["date"].map(daily_market_index_medians)
    df["market_index"] = df["market_index"].fillna(date_impute).fillna(global_market_median)

    # 4. Temporal parsing
    if "date" in df:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date_dt"].dt.year
        df["month"] = df["date_dt"].dt.month
        df["day"] = df["date_dt"].dt.day
        df["dayofweek"] = df["date_dt"].dt.dayofweek
        df["dayofyear"] = df["date_dt"].dt.dayofyear
        df["weekofyear"] = df["date_dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["is_month_end"] = (df["day"] >= 27).astype(int)

    metadata = {
        "city_coords": city_coords,
        "equipment_weight_medians": equipment_weight_medians,
        "daily_market_index_medians": daily_market_index_medians,
        "global_market_median": global_market_median,
    }
    return df, metadata
