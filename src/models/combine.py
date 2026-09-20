"""Combine frequency and severity models into expected loss, and validate the
combination against actual aggregate loss on the test set.

expected_loss = P(claim) * E[severity | claim]

Uses the CALIBRATED frequency probability (not the raw scale_pos_weight-inflated
score -- see src/models/frequency.py's fit_calibrator docstring) and the
smearing-corrected severity prediction (see src/models/severity.py's
compute_smearing_factor docstring), since both raw model outputs are individually
biased in ways that would compound in the product.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REPORTS_FIGURES_DIR, REPORTS_METRICS_DIR


def compute_expected_loss(
    calibrated_p_claim: np.ndarray,
    smeared_expected_severity: np.ndarray,
    payout_given_claim_rate: float = 1.0,
) -> np.ndarray:
    """expected_loss = P(claim) * P(payout | claim) * E[severity | payout]

    payout_given_claim_rate corrects a real mismatch between what the two models
    were each trained to predict: the frequency model's target is HasClaim
    (ClaimNb > 0), which is the actuarially correct notion of "a claim occurred" --
    but ~27% of policies with ClaimNb > 0 have no matching severity record at all
    (see src/data/merge_clean.py's docstring on this dataset's known freq/sev
    extraction mismatch). The severity model, trained only on ClaimAmountTotal > 0
    rows, predicts E[severity | a real payout exists], not E[severity | ClaimNb > 0].
    Multiplying P(ClaimNb > 0) directly by that severity estimate overstates
    expected loss by roughly 1 / payout_given_claim_rate, since it implicitly
    assumes every counted claim has a real payout. payout_given_claim_rate should be
    estimated on the TRAIN split as (rows with ClaimAmountTotal > 0) / (rows with
    ClaimNb > 0) and passed in explicitly -- defaulting to 1.0 here (no correction)
    so callers must opt in deliberately rather than silently getting a biased
    combination.
    """
    return calibrated_p_claim * payout_given_claim_rate * smeared_expected_severity


def compute_payout_given_claim_rate(train_df: pd.DataFrame) -> float:
    claiming = train_df["HasClaim"] == 1
    return float((train_df.loc[claiming, "ClaimAmountTotal"] > 0).mean())


def validate_aggregate_loss(expected_loss: pd.Series, actual_claim_amount: pd.Series) -> dict:
    total_predicted = float(expected_loss.sum())
    total_actual = float(actual_claim_amount.sum())
    return {
        "total_predicted": total_predicted,
        "total_actual": total_actual,
        "ratio_predicted_to_actual": total_predicted / total_actual,
    }


def decile_lift_chart(
    expected_loss: pd.Series,
    actual_claim_amount: pd.Series,
    out_path: Path = REPORTS_FIGURES_DIR / "decile_lift_chart.png",
) -> pd.DataFrame:
    """Bucket policies into deciles by predicted expected loss and compare mean
    predicted vs mean actual per decile -- a well-behaved model should show
    monotonically increasing actual loss across deciles, since that's what "the
    model correctly ranks risk" looks like in aggregate, even though per-policy
    expected loss is a noisy, low-signal prediction (any individual claim is mostly
    driven by chance, not the policy's risk factors).
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"expected_loss": expected_loss, "actual": actual_claim_amount})
    df["decile"] = pd.qcut(df["expected_loss"], 10, labels=False, duplicates="drop")
    decile_summary = df.groupby("decile")[["expected_loss", "actual"]].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = decile_summary.index + 1
    width = 0.35
    ax.bar(x - width / 2, decile_summary["expected_loss"], width, label="Mean predicted expected loss", color="#4C72B0")
    ax.bar(x + width / 2, decile_summary["actual"], width, label="Mean actual loss", color="#DD8452")
    ax.set_xlabel("Decile of predicted expected loss (1=lowest risk, 10=highest)")
    ax.set_ylabel("Mean loss per policy (EUR)")
    ax.set_title("Decile lift chart: predicted vs actual loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return decile_summary


def save_combination_artifacts(
    validation: dict,
    metrics_path: Path = REPORTS_METRICS_DIR / "combined_validation.json",
) -> None:
    metrics_path.write_text(json.dumps(validation, indent=2))
