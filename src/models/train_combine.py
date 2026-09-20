"""Entry point: combine frequency and severity predictions into expected loss on
the full test set (both claim and non-claim policies) and validate against actual
aggregate loss.

Run as `python -m src.models.train_combine` after train_frequency.py and
train_severity.py.
"""

import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.features.engineer import build_feature_matrix
from src.models.combine import (
    compute_expected_loss,
    compute_payout_given_claim_rate,
    decile_lift_chart,
    save_combination_artifacts,
    validate_aggregate_loss,
)
from src.models.split import load_split_ids


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "merged_clean.parquet")
    ids = load_split_ids()
    train = df[df["IDpol"].isin(ids["train"])]
    test = df[df["IDpol"].isin(ids["test"])].copy()

    # Estimated on TRAIN only, matching the same leakage-avoidance rule used for
    # scale_pos_weight -- see compute_expected_loss's docstring for why this
    # correction is needed at all.
    payout_given_claim_rate = compute_payout_given_claim_rate(train)
    print(f"P(real payout | ClaimNb>0), train-estimated: {payout_given_claim_rate:.4f}")

    encoders = joblib.load(MODELS_DIR / "feature_encoders.joblib")
    X_test, _ = build_feature_matrix(test, fit_encoders=False, encoders=encoders)

    freq_model = XGBClassifier()
    freq_model.load_model(str(MODELS_DIR / "frequency_model.json"))
    calibrator = joblib.load(MODELS_DIR / "frequency_calibrator.joblib")

    sev_model = XGBRegressor()
    sev_model.load_model(str(MODELS_DIR / "severity_model.json"))
    smearing_factor = json.loads((MODELS_DIR / "severity_smearing_factor.json").read_text())["smearing_factor"]

    raw_p_claim = freq_model.predict_proba(X_test)[:, 1]
    calibrated_p_claim = calibrator.predict(raw_p_claim)

    # Severity is predicted for EVERY test policy (not just claimants) since expected
    # loss needs an E[severity|claim] estimate even for policies that didn't claim --
    # the frequency term is what correctly drives their expected loss toward zero.
    pred_severity_log = sev_model.predict(X_test)
    expected_severity = np.expm1(pred_severity_log) * smearing_factor

    expected_loss = compute_expected_loss(calibrated_p_claim, expected_severity, payout_given_claim_rate)
    expected_loss_series = pd.Series(expected_loss, index=test.index)

    validation = validate_aggregate_loss(expected_loss_series, test["ClaimAmountTotal"])
    print("Aggregate validation:", validation)

    decile_summary = decile_lift_chart(expected_loss_series, test["ClaimAmountTotal"])
    print(decile_summary)

    save_combination_artifacts(validation)
    print("Combination artifacts saved.")


if __name__ == "__main__":
    main()
