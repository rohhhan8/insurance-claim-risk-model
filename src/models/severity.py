"""Severity model: XGBRegressor predicting E[severity | claim], trained only on rows
with a confirmed claim (ClaimNb > 0, ClaimAmountTotal > 0).

Target is log1p(ClaimAmountTotal), fit with reg:squarederror -- not reg:tweedie.
Tweedie is the right choice when a SINGLE model must jointly handle the zero-mass
and the continuous positive tail (i.e. modeling pure premium directly, without
decomposing into frequency x severity). This pipeline already separates the two, so
the zero-inflation problem Tweedie solves has already been handled by the frequency
classifier -- applying Tweedie to a zero-free subset would be redundant, since its
power parameter is tuned to model a zero-inflated distribution that no longer exists
here. The EDA notebook shows raw severity skew ~106 dropping to ~-0.5 after log1p,
which is close enough to symmetric that squared-error loss on the log scale is a
reasonable, standard fit objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.config import MODELS_DIR, RANDOM_SEED, REPORTS_METRICS_DIR

GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.7, "colsample_bytree": 0.7},
    {"max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 1.0},
]


def filter_to_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real recorded payout only -- see merge_clean.py's docstring on
    the ~27% of ClaimNb>0 rows with no severity record: those rows have no true
    target value to train or evaluate against, so they're excluded here rather than
    treated as ClaimAmountTotal=0 (which would teach the model that many genuine
    claims cost nothing, dragging predictions down for no real reason)."""
    return df.loc[df["ClaimAmountTotal"] > 0].copy()


def prepare_severity_target(df: pd.DataFrame) -> pd.Series:
    return np.log1p(df["ClaimAmountTotal"])


def run_grid_search(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_val: pd.DataFrame,
    y_val_log: pd.Series,
    grid: list[dict] = GRID,
    seed: int = RANDOM_SEED,
) -> tuple[XGBRegressor, pd.DataFrame]:
    results = []
    best_model = None
    best_rmse = np.inf

    for params in grid:
        model = XGBRegressor(
            n_estimators=1000,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=30,
            random_state=seed,
            **params,
        )
        model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
        val_pred_log = model.predict(X_val)
        rmse_log = mean_squared_error(y_val_log, val_pred_log) ** 0.5
        results.append({**params, "best_iteration": model.best_iteration, "val_rmse_log": rmse_log})

        if rmse_log < best_rmse:
            best_rmse = rmse_log
            best_model = model

    results_df = pd.DataFrame(results).sort_values("val_rmse_log")
    return best_model, results_df


def compute_smearing_factor(model: XGBRegressor, X_val: pd.DataFrame, y_val_log: pd.Series) -> float:
    """Duan's smearing estimator: mean(exp(residual)) on validation, used to correct
    the systematic downward bias of exp(predicted-log-mean) as an estimator of the
    true mean under log-skewed data (Jensen's inequality -- E[exp(X)] > exp(E[X])).
    Fit on validation, not train, so the correction isn't overfit to the same rows
    the tree itself was fit on.
    """
    val_pred_log = model.predict(X_val)
    residuals = y_val_log - val_pred_log
    return float(np.mean(np.exp(residuals)))


def evaluate_severity(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test_actual: pd.Series,
    smearing_factor: float,
) -> dict:
    """RMSE/MAE on the ORIGINAL scale (not log scale), overall and on the top 10% of
    claims by actual amount -- the tail is where an insurer's real cost exposure is,
    so a model that's great on typical claims but bad on large ones is understating
    the risk that matters most.
    """
    pred_log = model.predict(X_test)
    pred_original = np.expm1(pred_log) * smearing_factor

    rmse = mean_squared_error(y_test_actual, pred_original) ** 0.5
    mae = mean_absolute_error(y_test_actual, pred_original)

    top_10pct_threshold = y_test_actual.quantile(0.9)
    top_mask = y_test_actual >= top_10pct_threshold
    rmse_top10 = mean_squared_error(y_test_actual[top_mask], pred_original[top_mask]) ** 0.5
    mae_top10 = mean_absolute_error(y_test_actual[top_mask], pred_original[top_mask])

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "top_10pct_threshold": float(top_10pct_threshold),
        "rmse_top_10pct": float(rmse_top10),
        "mae_top_10pct": float(mae_top10),
        "smearing_factor": smearing_factor,
    }


def save_severity_artifacts(
    model: XGBRegressor,
    smearing_factor: float,
    metrics: dict,
    grid_results: pd.DataFrame,
    model_path: Path = MODELS_DIR / "severity_model.json",
    smearing_path: Path = MODELS_DIR / "severity_smearing_factor.json",
    metrics_path: Path = REPORTS_METRICS_DIR / "severity_metrics.json",
    grid_path: Path = REPORTS_METRICS_DIR / "severity_grid_search.csv",
) -> None:
    model.save_model(str(model_path))
    smearing_path.write_text(json.dumps({"smearing_factor": smearing_factor}))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    grid_results.to_csv(grid_path, index=False)
