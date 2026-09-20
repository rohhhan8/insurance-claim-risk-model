"""Entry point: run SHAP analysis for both models and save summary plots + top
feature rankings. Run as `python -m src.interpret.run_shap` after both models are
trained.
"""

import json

import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_METRICS_DIR
from src.features.engineer import build_feature_matrix
from src.interpret.shap_explain import (
    explain_frequency,
    explain_severity,
    sample_for_shap,
    top_features_by_mean_abs_shap,
)
from src.models.severity import filter_to_claims
from src.models.split import load_split_ids


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "merged_clean.parquet")
    ids = load_split_ids()
    test = df[df["IDpol"].isin(ids["test"])].copy()

    encoders = joblib.load(MODELS_DIR / "feature_encoders.joblib")

    # Frequency SHAP: sampled from the full test set (every policy has a P(claim)).
    X_test_freq, _ = build_feature_matrix(test, fit_encoders=False, encoders=encoders)
    X_freq_sample = sample_for_shap(X_test_freq)

    freq_model = XGBClassifier()
    freq_model.load_model(str(MODELS_DIR / "frequency_model.json"))
    freq_shap_values = explain_frequency(freq_model, X_freq_sample)
    freq_top = top_features_by_mean_abs_shap(freq_shap_values, X_freq_sample.columns.tolist())
    print("Top frequency features by mean |SHAP|:\n", freq_top)

    # Severity SHAP: sampled from claiming rows only, matching how the model trained.
    test_claims = filter_to_claims(test)
    X_test_sev, _ = build_feature_matrix(test_claims, fit_encoders=False, encoders=encoders)
    X_sev_sample = sample_for_shap(X_test_sev)

    sev_model = XGBRegressor()
    sev_model.load_model(str(MODELS_DIR / "severity_model.json"))
    sev_shap_values = explain_severity(sev_model, X_sev_sample)
    sev_top = top_features_by_mean_abs_shap(sev_shap_values, X_sev_sample.columns.tolist())
    print("Top severity features by mean |SHAP|:\n", sev_top)

    comparison = {
        "frequency_top_features": freq_top.to_dict(),
        "severity_top_features": sev_top.to_dict(),
    }
    (REPORTS_METRICS_DIR / "shap_top_features.json").write_text(json.dumps(comparison, indent=2))
    print("SHAP artifacts saved.")


if __name__ == "__main__":
    main()
