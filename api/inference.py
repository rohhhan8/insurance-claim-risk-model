"""Model loading and prediction logic for the API.

Reuses src/features/engineer.py's build_feature_matrix directly rather than
reimplementing the transform here -- duplicating that logic would risk train/serve
skew if the two copies ever drifted (e.g. a new derived feature added to training
but forgotten in the API).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier, XGBRegressor

from src.config import MODELS_DIR
from src.features.engineer import build_feature_matrix

from api.schemas import PolicyFeatures, PredictionResponse, ShapContribution

TOP_N_SHAP = 5


class ModelBundle:
    """Holds everything loaded once at API startup -- models, encoders, calibrator,
    smearing/payout correction factors, and SHAP explainers."""

    def __init__(self) -> None:
        self.freq_model = XGBClassifier()
        self.freq_model.load_model(str(MODELS_DIR / "frequency_model.json"))

        self.sev_model = XGBRegressor()
        self.sev_model.load_model(str(MODELS_DIR / "severity_model.json"))

        self.calibrator = joblib.load(MODELS_DIR / "frequency_calibrator.joblib")
        self.encoders = joblib.load(MODELS_DIR / "feature_encoders.joblib")
        self.feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")

        self.smearing_factor = json.loads(
            (MODELS_DIR / "severity_smearing_factor.json").read_text()
        )["smearing_factor"]
        self.payout_given_claim_rate = json.loads(
            (MODELS_DIR / "payout_given_claim_rate.json").read_text()
        )["payout_given_claim_rate"]

        self.freq_explainer = shap.TreeExplainer(self.freq_model)
        self.sev_explainer = shap.TreeExplainer(self.sev_model)


def load_models() -> ModelBundle:
    return ModelBundle()


def transform_input(payload: PolicyFeatures, bundle: ModelBundle) -> pd.DataFrame:
    raw_df = pd.DataFrame([payload.model_dump()])
    feature_matrix, _ = build_feature_matrix(raw_df, fit_encoders=False, encoders=bundle.encoders)
    # Reindex to the exact training column order/set -- a category unseen at fit
    # time is dropped by handle_unknown="ignore", not added, so this only ever
    # fills genuinely absent columns (there shouldn't be any) rather than masking
    # a real mismatch.
    return feature_matrix.reindex(columns=bundle.feature_columns, fill_value=0.0)


def _top_shap_contributions(
    shap_values: np.ndarray, feature_names: list[str], top_n: int = TOP_N_SHAP
) -> list[ShapContribution]:
    row = shap_values[0]
    order = np.argsort(-np.abs(row))[:top_n]
    return [ShapContribution(feature=feature_names[i], contribution=float(row[i])) for i in order]


def predict_single(payload: PolicyFeatures, bundle: ModelBundle) -> PredictionResponse:
    X = transform_input(payload, bundle)

    raw_p_claim = bundle.freq_model.predict_proba(X)[:, 1]
    calibrated_p_claim = float(bundle.calibrator.predict(raw_p_claim)[0])

    pred_severity_log = bundle.sev_model.predict(X)
    expected_severity = float(np.expm1(pred_severity_log)[0] * bundle.smearing_factor)

    expected_loss = calibrated_p_claim * bundle.payout_given_claim_rate * expected_severity

    freq_shap = bundle.freq_explainer.shap_values(X)
    sev_shap = bundle.sev_explainer.shap_values(X)

    return PredictionResponse(
        probability_of_claim=calibrated_p_claim,
        expected_severity=expected_severity,
        expected_loss=expected_loss,
        top_shap_frequency=_top_shap_contributions(freq_shap, X.columns.tolist()),
        top_shap_severity=_top_shap_contributions(sev_shap, X.columns.tolist()),
    )
