"""SHAP TreeExplainer analysis for the frequency and severity models, kept as two
fully separate explainers -- never share one TreeExplainer across models, since each
is fit against a different tree ensemble and background distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier, XGBRegressor

from src.config import RANDOM_SEED, REPORTS_FIGURES_DIR

SHAP_SAMPLE_SIZE = 5000  # full test set is unnecessarily slow for TreeExplainer on a
# laptop, and 5,000 rows is enough for a stable summary-plot ranking.


def sample_for_shap(X: pd.DataFrame, n: int = SHAP_SAMPLE_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=seed)


def explain_frequency(
    model: XGBClassifier, X_sample: pd.DataFrame, out_path: Path = REPORTS_FIGURES_DIR / "frequency_shap_summary.png"
) -> np.ndarray:
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP summary: frequency model (P(claim))")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return shap_values


def explain_severity(
    model: XGBRegressor, X_sample: pd.DataFrame, out_path: Path = REPORTS_FIGURES_DIR / "severity_shap_summary.png"
) -> np.ndarray:
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP summary: severity model (log severity | claim)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return shap_values


def top_features_by_mean_abs_shap(shap_values: np.ndarray, feature_names: list[str], top_n: int = 5) -> pd.Series:
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    return ranked.head(top_n)
