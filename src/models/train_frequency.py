"""Entry point: train the frequency model end to end and save all artifacts.

Run as `python -m src.models.train_frequency` from the project root (with the venv
active). Writes the split IDs, feature encoders, trained model, isotonic calibrator,
metrics, and the calibration plot -- everything the severity stage and the API need.
"""

import joblib
import pandas as pd

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.features.engineer import build_feature_matrix
from src.models.frequency import (
    compute_scale_pos_weight,
    evaluate_frequency,
    fit_calibrator,
    plot_calibration_curve,
    run_grid_search,
    save_frequency_artifacts,
)
from src.models.split import save_split_ids, split_data


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "merged_clean.parquet")
    train, val, test = split_data(df)
    save_split_ids(train, val, test)

    X_train, encoders = build_feature_matrix(train, fit_encoders=True)
    X_val, _ = build_feature_matrix(val, fit_encoders=False, encoders=encoders)
    X_test, _ = build_feature_matrix(test, fit_encoders=False, encoders=encoders)
    y_train, y_val, y_test = train["HasClaim"], val["HasClaim"], test["HasClaim"]

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight (train-only): {scale_pos_weight:.4f}")

    best_model, grid_results = run_grid_search(X_train, y_train, X_val, y_val, scale_pos_weight)
    print(grid_results)

    calibrator = fit_calibrator(best_model, X_val, y_val)

    metrics = evaluate_frequency(best_model, X_test, y_test, calibrator=calibrator)
    print("Test metrics:", metrics)

    raw_test_probs = best_model.predict_proba(X_test)[:, 1]
    calibrated_test_probs = calibrator.predict(raw_test_probs)
    plot_calibration_curve(y_test, raw_test_probs, calibrated_test_probs)

    save_frequency_artifacts(best_model, calibrator, metrics, grid_results)
    joblib.dump(encoders, MODELS_DIR / "feature_encoders.joblib")
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_columns.joblib")
    print("Frequency model artifacts saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
