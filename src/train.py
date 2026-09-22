"""
Model training and temporal validation pipeline.
Evaluates Ridge, LightGBM, XGBoost, CatBoost, and an Ensemble on an out-of-time split.
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from src.data_loader import load_dataset, clean_and_prepare
from src.features import engineer_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS


def train_and_validate(
    train_path: str = "data/train_test.csv",
    val_path: str = "data/validation.csv",
    models_dir: str = "models",
) -> dict:
    """
    Executes the training and validation workflow.
    """
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPOTTER FREIGHT RATE PREDICTION - MODEL TRAINING & VALIDATION")
    print("=" * 70)

    # 1. Load data
    train_raw = load_dataset(train_path)
    val_raw = load_dataset(val_path) if Path(val_path).exists() else None

    # Clean train dataset and obtain metadata
    train_cleaned, metadata = clean_and_prepare(train_raw)

    # If validation data is present, also augment metadata with December dates from val
    if val_raw is not None:
        val_cleaned, val_meta = clean_and_prepare(
            val_raw,
            city_coords=metadata["city_coords"],
            equipment_weight_medians=metadata["equipment_weight_medians"],
            daily_market_index_medians=metadata["daily_market_index_medians"],
        )
        # Update daily market index medians with validation dates (Nov-Dec)
        for d, m in val_meta["daily_market_index_medians"].items():
            metadata["daily_market_index_medians"][d] = m

    # Feature engineering on train
    train_feat, feat_meta = engineer_features(train_cleaned, is_training=True)

    # 2. Out-of-Time Validation Split
    # Train: Months 1-8 (Jan - Aug 2025: 38,477 loads)
    # Val: Months 9-10 (Sep - Oct 2025: 9,523 loads, 2 full months)
    cutoff_date = "2025-09-01"
    train_split = train_feat[train_feat["date_dt"] < cutoff_date].copy()
    val_split = train_feat[train_feat["date_dt"] >= cutoff_date].copy()

    print(f"\n[Data Split] Chronological Out-of-Time Split at {cutoff_date}:")
    print(f"  Training Horizon:   {train_split['date'].min()} to {train_split['date'].max()} ({len(train_split):,} loads)")
    print(f"  Validation Horizon: {val_split['date'].min()} to {val_split['date'].max()} ({len(val_split):,} loads)")

    # Prepare feature matrices
    X_train_num = train_split[FEATURE_COLUMNS].copy()
    X_val_num = val_split[FEATURE_COLUMNS].copy()
    y_train = train_split["posted_rate"].values
    y_val = val_split["posted_rate"].values

    # Category mappings for tree algorithms
    for col in CATEGORICAL_COLUMNS:
        train_split[col] = train_split[col].astype("category")
        val_split[col] = val_split[col].astype("category")

    all_features = FEATURE_COLUMNS + CATEGORICAL_COLUMNS
    X_train_all = train_split[all_features].copy()
    X_val_all = val_split[all_features].copy()

    # Model dictionary for evaluation
    benchmark_results = {}

    # Benchmark 1: Naive Quote Baseline (distance * quote_signal)
    base_preds = val_split["base_quote"].values
    benchmark_results["Naive Quote Baseline"] = {
        "MAE": float(mean_absolute_error(y_val, base_preds)),
        "RMSE": float(root_mean_squared_error(y_val, base_preds)),
        "R2": float(r2_score(y_val, base_preds)),
        "MAPE": float(mean_absolute_percentage_error(y_val, base_preds)),
    }

    # Benchmark 2: Ridge Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_val_scaled = scaler.transform(X_val_num)

    ridge = Ridge(alpha=100.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    ridge_preds = np.clip(ridge.predict(X_val_scaled), 50.0, None)
    benchmark_results["Ridge Regression"] = {
        "MAE": float(mean_absolute_error(y_val, ridge_preds)),
        "RMSE": float(root_mean_squared_error(y_val, ridge_preds)),
        "R2": float(r2_score(y_val, ridge_preds)),
        "MAPE": float(mean_absolute_percentage_error(y_val, ridge_preds)),
    }

    # Benchmark 3: LightGBM Regressor
    print("\nTraining LightGBM Regressor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=750,
        learning_rate=0.035,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train_all, y_train)
    lgb_preds = np.clip(lgb_model.predict(X_val_all), 50.0, None)
    benchmark_results["LightGBM"] = {
        "MAE": float(mean_absolute_error(y_val, lgb_preds)),
        "RMSE": float(root_mean_squared_error(y_val, lgb_preds)),
        "R2": float(r2_score(y_val, lgb_preds)),
        "MAPE": float(mean_absolute_percentage_error(y_val, lgb_preds)),
    }

    # Benchmark 4: XGBoost Regressor
    print("Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=650,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_all, y_train)
    xgb_preds = np.clip(xgb_model.predict(X_val_all), 50.0, None)
    benchmark_results["XGBoost"] = {
        "MAE": float(mean_absolute_error(y_val, xgb_preds)),
        "RMSE": float(root_mean_squared_error(y_val, xgb_preds)),
        "R2": float(r2_score(y_val, xgb_preds)),
        "MAPE": float(mean_absolute_percentage_error(y_val, xgb_preds)),
    }

    # Benchmark 5: CatBoost Regressor
    print("Training CatBoost Regressor...")
    cat_model = CatBoostRegressor(
        iterations=700,
        learning_rate=0.04,
        depth=6,
        cat_features=CATEGORICAL_COLUMNS,
        random_seed=42,
        verbose=False,
    )
    cat_model.fit(X_train_all, y_train)
    cat_preds = np.clip(cat_model.predict(X_val_all), 50.0, None)
    benchmark_results["CatBoost"] = {
        "MAE": float(mean_absolute_error(y_val, cat_preds)),
        "RMSE": float(root_mean_squared_error(y_val, cat_preds)),
        "R2": float(r2_score(y_val, cat_preds)),
        "MAPE": float(mean_absolute_percentage_error(y_val, cat_preds)),
    }

    # Benchmark 6: Blended Ensemble (LightGBM + XGBoost + CatBoost)
    # Finding optimal blend weights via constrained grid optimization
    best_mae = float("inf")
    best_weights = (0.45, 0.35, 0.20)
    for w_lgb in np.linspace(0.2, 0.6, 9):
        for w_xgb in np.linspace(0.2, 0.6, 9):
            w_cat = 1.0 - w_lgb - w_xgb
            if w_cat < 0.1 or w_cat > 0.5:
                continue
            blend_candidate = w_lgb * lgb_preds + w_xgb * xgb_preds + w_cat * cat_preds
            mae_candidate = mean_absolute_error(y_val, blend_candidate)
            if mae_candidate < best_mae:
                best_mae = mae_candidate
                best_weights = (float(w_lgb), float(w_xgb), float(w_cat))

    ens_preds = best_weights[0] * lgb_preds + best_weights[1] * xgb_preds + best_weights[2] * cat_preds
    benchmark_results["Blended Ensemble"] = {
        "MAE": float(mean_absolute_error(y_val, ens_preds)),
        "RMSE": float(root_mean_squared_error(y_val, ens_preds)),
        "R2": float(r2_score(y_val, ens_preds)),
        "MAPE": float(mean_absolute_percentage_error(y_val, ens_preds)),
    }

    # Display Benchmark Table
    print("\n" + "=" * 70)
    print("HOLD-OUT VALIDATION BENCHMARK RESULTS (Months 9-10: Sep-Oct 2025)")
    print("=" * 70)
    print(f"{'Model Name':<24} | {'MAE ($)':<10} | {'RMSE ($)':<10} | {'R2 Score':<10} | {'MAPE':<10}")
    print("-" * 70)
    for name, m in benchmark_results.items():
        print(f"{name:<24} | {m['MAE']:<10.2f} | {m['RMSE']:<10.2f} | {m['R2']:<10.4f} | {m['MAPE']*100:<9.2f}%")
    print("=" * 70)

    # 3. Train Final Production Models on 100% of Labeled Data
    print("\nTraining Final Production Models on full development dataset (48,000 loads)...")
    for col in CATEGORICAL_COLUMNS:
        train_feat[col] = train_feat[col].astype("category")

    X_full = train_feat[all_features].copy()
    y_full = train_feat["posted_rate"].values

    final_lgb = lgb.LGBMRegressor(
        n_estimators=850,
        learning_rate=0.035,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final_lgb.fit(X_full, y_full)

    final_xgb = xgb.XGBRegressor(
        n_estimators=750,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )
    final_xgb.fit(X_full, y_full)

    final_cat = CatBoostRegressor(
        iterations=800,
        learning_rate=0.04,
        depth=6,
        cat_features=CATEGORICAL_COLUMNS,
        random_seed=42,
        verbose=False,
    )
    final_cat.fit(X_full, y_full)

    # Save artifacts
    artifacts = {
        "final_lgb": final_lgb,
        "final_xgb": final_xgb,
        "final_cat": final_cat,
        "best_weights": best_weights,
        "metadata": metadata,
        "feat_meta": feat_meta,
        "all_features": all_features,
        "benchmark_results": benchmark_results,
    }

    with open(out_dir / "models.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nSaved models and metadata to {out_dir / 'models.pkl'}")
    return artifacts


if __name__ == "__main__":
    train_and_validate()
