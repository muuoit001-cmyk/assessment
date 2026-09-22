"""
Prediction inference script for Freight Rate Prediction.
Generates predictions for validation.csv and december_chart_inputs.csv.
"""

from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_dataset, clean_and_prepare
from src.features import engineer_features, CATEGORICAL_COLUMNS


def run_predictions(
    models_path: str = "models/models.pkl",
    val_path: str = "data/validation.csv",
    val_template_path: str = "data/validation_predictions_template.csv",
    val_output_path: str = "validation_predictions.csv",
    dec_path: str = "data/december_chart_inputs.csv",
) -> None:
    """
    Loads saved models, produces validation predictions and December scenario predictions.
    """
    print("=" * 70)
    print("RUNNING INFERENCE PIPELINE")
    print("=" * 70)

    with open(models_path, "rb") as f:
        bundle = pickle.load(f)

    final_lgb = bundle["final_lgb"]
    final_xgb = bundle["final_xgb"]
    final_cat = bundle["final_cat"]
    best_weights = bundle["best_weights"]
    metadata = bundle["metadata"]
    feat_meta = bundle["feat_meta"]
    all_features = bundle["all_features"]

    print(f"Loaded models from {models_path}")
    print(f"Ensemble weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")

    # =========================================================================
    # 1. Validation Set Predictions (12,000 loads)
    # =========================================================================
    print(f"\n[1/2] Processing validation set from {val_path}...")
    val_raw = load_dataset(val_path)
    val_cleaned, _ = clean_and_prepare(
        val_raw,
        city_coords=metadata["city_coords"],
        equipment_weight_medians=metadata["equipment_weight_medians"],
        daily_market_index_medians=metadata["daily_market_index_medians"],
        global_market_median=metadata["global_market_median"],
    )
    val_feat, _ = engineer_features(
        val_cleaned,
        lane_stats=feat_meta["lane_stats"],
        equipment_quote_medians=feat_meta["equipment_quote_medians"],
        is_training=False,
    )

    for col in CATEGORICAL_COLUMNS:
        val_feat[col] = val_feat[col].astype("category")

    X_val = val_feat[all_features].copy()

    # Predict with each model
    p_lgb = final_lgb.predict(X_val)
    p_xgb = final_xgb.predict(X_val)
    p_cat = final_cat.predict(X_val)

    # Ensemble blend
    p_ens = best_weights[0] * p_lgb + best_weights[1] * p_xgb + best_weights[2] * p_cat
    p_ens = np.clip(p_ens, 50.0, None)  # Ensure positive valid spot rates
    p_ens = np.round(p_ens, 2)

    val_feat["predicted_rate"] = p_ens

    # Match template
    val_template = load_dataset(val_template_path)
    pred_map = dict(zip(val_feat["load_id"], val_feat["predicted_rate"]))
    val_template["predicted_rate"] = val_template["load_id"].map(pred_map)

    # Validate formatting
    assert len(val_template) == 12_000, f"Expected 12,000 rows, got {len(val_template)}"
    assert not val_template["predicted_rate"].isna().any(), "Found missing predicted_rate values!"
    assert (val_template["predicted_rate"] > 0).all(), "Found non-positive predicted_rate values!"
    assert list(val_template.columns) == ["load_id", "predicted_rate"], "Columns mismatch!"

    # Save validation_predictions.csv in both root and data/
    val_template.to_csv(val_output_path, index=False)
    val_template.to_csv("data/validation_predictions.csv", index=False)
    print(f" Saved 12,000 validation predictions to {val_output_path}")

    # =========================================================================
    # 2. December Inputs Predictions (31 loads)
    # =========================================================================
    print(f"\n[2/2] Processing December inputs from {dec_path}...")
    dec_raw = load_dataset(dec_path)
    dec_cleaned, _ = clean_and_prepare(
        dec_raw,
        city_coords=metadata["city_coords"],
        equipment_weight_medians=metadata["equipment_weight_medians"],
        daily_market_index_medians=metadata["daily_market_index_medians"],
        global_market_median=metadata["global_market_median"],
    )
    dec_feat, _ = engineer_features(
        dec_cleaned,
        lane_stats=feat_meta["lane_stats"],
        equipment_quote_medians=feat_meta["equipment_quote_medians"],
        is_training=False,
    )

    for col in CATEGORICAL_COLUMNS:
        dec_feat[col] = dec_feat[col].astype("category")

    X_dec = dec_feat[all_features].copy()

    d_lgb = final_lgb.predict(X_dec)
    d_xgb = final_xgb.predict(X_dec)
    d_cat = final_cat.predict(X_dec)

    d_ens = best_weights[0] * d_lgb + best_weights[1] * d_xgb + best_weights[2] * d_cat
    d_ens = np.clip(d_ens, 50.0, None)
    d_ens = np.round(d_ens, 2)

    dec_raw["predicted_rate"] = d_ens

    # Verify column order and non-null
    cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dec_raw = dec_raw[cols]
    assert len(dec_raw) == 31, f"Expected 31 rows, got {len(dec_raw)}"
    assert not dec_raw["predicted_rate"].isna().any(), "Found NaN in December predicted_rate"
    assert (dec_raw["predicted_rate"] > 0).all(), "Found non-positive December rates"

    # Save to both locations
    dec_raw.to_csv(dec_path, index=False)
    dec_raw.to_csv("december-chart-inputs.csv", index=False)
    print(f" Saved 31 December predictions to {dec_path} and december-chart-inputs.csv")
    print(f"  December Predicted Rate Range: min=${d_ens.min():.2f}, mean=${d_ens.mean():.2f}, max=${d_ens.max():.2f}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    run_predictions()
