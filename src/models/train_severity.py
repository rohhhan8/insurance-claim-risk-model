"""Entry point: train the severity model end to end and save all artifacts.

Run as `python -m src.models.train_severity` after train_frequency.py -- it reuses
the same IDpol split (via split_ids.json) so frequency and severity predictions on
the test set line up for the Step 5 expected-loss combination.
"""

import joblib
import pandas as pd

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.features.engineer import build_feature_matrix
from src.models.severity import (
    compute_smearing_factor,
    evaluate_severity,
    filter_to_claims,
    prepare_severity_target,
    run_grid_search,
    save_severity_artifacts,
)
from src.models.split import load_split_ids


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "merged_clean.parquet")
    ids = load_split_ids()

    train = df[df["IDpol"].isin(ids["train"])]
    val = df[df["IDpol"].isin(ids["val"])]
    test = df[df["IDpol"].isin(ids["test"])]

    # Filter to claiming rows AFTER reusing the frequency split, not before --
    # preserves the same train/val/test IDpol membership across both stages.
    train_claims = filter_to_claims(train)
    val_claims = filter_to_claims(val)
    test_claims = filter_to_claims(test)
    print(f"claim rows: train={len(train_claims)} val={len(val_claims)} test={len(test_claims)}")

    # Reuse the SAME fitted encoders as the frequency model (persisted to disk by
    # train_frequency.py) rather than refitting on the claims-only subset -- fitting
    # fresh one-hot categories here could silently diverge from what the frequency
    # model and the API expect.
    encoders = joblib.load(MODELS_DIR / "feature_encoders.joblib")
    X_train, _ = build_feature_matrix(train_claims, fit_encoders=False, encoders=encoders)
    X_val, _ = build_feature_matrix(val_claims, fit_encoders=False, encoders=encoders)
    X_test, _ = build_feature_matrix(test_claims, fit_encoders=False, encoders=encoders)

    y_train_log = prepare_severity_target(train_claims)
    y_val_log = prepare_severity_target(val_claims)
    y_test_actual = test_claims["ClaimAmountTotal"]

    best_model, grid_results = run_grid_search(X_train, y_train_log, X_val, y_val_log)
    print(grid_results)

    smearing_factor = compute_smearing_factor(best_model, X_val, y_val_log)
    print(f"Duan smearing factor: {smearing_factor:.4f}")

    metrics = evaluate_severity(best_model, X_test, y_test_actual, smearing_factor)
    print("Test metrics:", metrics)

    save_severity_artifacts(best_model, smearing_factor, metrics, grid_results)
    print("Severity model artifacts saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
