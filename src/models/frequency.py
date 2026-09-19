"""Frequency model: XGBClassifier predicting P(claim) per policy.

Target is binary HasClaim (ClaimNb > 0), not a Poisson count. A count model is more
standard actuarial practice in general, but the expected-loss combination this
project builds toward is `P(claim) * E[severity|claim]`, which needs a probability
-- and PR-AUC/calibration (the specified evaluation metrics) are only well-defined
for a binary target. The binary formulation is chosen to match that architecture,
not because it's the more "correct" actuarial choice in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.config import MODELS_DIR, RANDOM_SEED, REPORTS_FIGURES_DIR, REPORTS_METRICS_DIR

GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.7, "colsample_bytree": 0.7},
    {"max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 1.0},
]


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """neg/pos ratio on TRAIN ONLY -- computing this on the full dataset (pre-split)
    would leak val/test class balance into a value that influences training."""
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    return neg / pos


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    grid: list[dict] = GRID,
    seed: int = RANDOM_SEED,
) -> tuple[XGBClassifier, pd.DataFrame]:
    """Plain loop over a small manual grid, not GridSearchCV.

    GridSearchCV's fit/predict wrapper makes it awkward to pass a per-fold eval_set
    for early stopping -- a plain loop keeps that explicit and is easier to read for
    a project this size (4 fits total).
    """
    results = []
    best_model = None
    best_score = -np.inf

    for params in grid:
        model = XGBClassifier(
            n_estimators=1000,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=30,
            random_state=seed,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_probs = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_probs)
        results.append({**params, "best_iteration": model.best_iteration, "val_pr_auc": pr_auc})

        if pr_auc > best_score:
            best_score = pr_auc
            best_model = model

    results_df = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False)
    return best_model, results_df


def fit_calibrator(model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> IsotonicRegression:
    """Recover calibrated probabilities from the scale_pos_weight-inflated scores.

    Fitting scale_pos_weight to handle the ~5% positive rate is what makes the raw
    model rankable and gives it a usable PR-AUC, but it also means predict_proba's
    output is not a real probability -- on this data the mean predicted probability
    comes out ~8.5x the actual claim rate. Isotonic regression fit on the VALIDATION
    set (never train, to avoid overfitting the calibration map to the same rows the
    tree was fit on) maps the raw score back onto the true probability scale, which
    is what src/models/combine.py needs for expected_loss = P(claim) * E[severity]
    to be comparable to actual aggregate loss.
    """
    raw_val_probs = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val_probs, y_val)
    return calibrator


def evaluate_frequency(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    calibrator: IsotonicRegression | None = None,
) -> dict:
    # PR-AUC is rank-based, so it's identical whether computed on raw or calibrated
    # probabilities -- isotonic regression is monotonic and doesn't change ranking.
    probs = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, probs)
    baseline_pr_auc = y_test.mean()  # PR-AUC of a random/no-skill classifier equals the positive rate

    metrics = {
        "pr_auc": float(pr_auc),
        "baseline_pr_auc": float(baseline_pr_auc),
        "test_claim_rate": float(y_test.mean()),
        "mean_raw_predicted_prob": float(probs.mean()),
    }
    if calibrator is not None:
        calibrated_probs = calibrator.predict(probs)
        metrics["mean_calibrated_predicted_prob"] = float(calibrated_probs.mean())
    return metrics


def plot_calibration_curve(
    y_true: pd.Series,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    out_path: Path = REPORTS_FIGURES_DIR / "frequency_calibration.png",
) -> None:
    """Plot both curves side by side: raw (scale_pos_weight-inflated, expected to
    sit well below the diagonal) and isotonic-calibrated (expected to hug it).

    The raw curve isn't a bug -- scale_pos_weight makes the score well-RANKED
    (higher predicted prob does mean higher observed frequency) but not well-
    CALIBRATED in absolute terms. The calibrated curve is what src/models/combine.py
    actually uses for expected_loss = P(claim) * E[severity].
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, raw_prob, n_bins=10, strategy="quantile")
    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", color="#C44E52", label="Raw (scale_pos_weight-inflated)")

    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, calibrated_prob, n_bins=10, strategy="quantile")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="o", color="#4C72B0", label="Isotonic-calibrated")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed claim frequency")
    ax.set_title("Calibration curve: raw vs isotonic-calibrated")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_frequency_artifacts(
    model: XGBClassifier,
    calibrator: IsotonicRegression,
    metrics: dict,
    grid_results: pd.DataFrame,
    model_path: Path = MODELS_DIR / "frequency_model.json",
    calibrator_path: Path = MODELS_DIR / "frequency_calibrator.joblib",
    metrics_path: Path = REPORTS_METRICS_DIR / "frequency_metrics.json",
    grid_path: Path = REPORTS_METRICS_DIR / "frequency_grid_search.csv",
) -> None:
    import joblib

    model.save_model(str(model_path))
    joblib.dump(calibrator, calibrator_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    grid_results.to_csv(grid_path, index=False)
