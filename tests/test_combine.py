import numpy as np
import pandas as pd

from src.models.combine import (
    compute_expected_loss,
    compute_payout_given_claim_rate,
    validate_aggregate_loss,
)


def test_compute_expected_loss_formula_elementwise():
    p_claim = np.array([0.1, 0.2, 0.05])
    severity = np.array([1000.0, 2000.0, 500.0])
    result = compute_expected_loss(p_claim, severity, payout_given_claim_rate=1.0)
    expected = p_claim * severity
    np.testing.assert_allclose(result, expected)


def test_compute_expected_loss_applies_payout_correction():
    p_claim = np.array([0.1])
    severity = np.array([1000.0])
    result = compute_expected_loss(p_claim, severity, payout_given_claim_rate=0.5)
    assert np.isclose(result[0], 0.1 * 0.5 * 1000.0)


def test_compute_payout_given_claim_rate():
    # 4 policies with a claim: 3 have a real payout, 1 doesn't (the dataset's known
    # freq/sev extraction mismatch) -> rate should be 0.75.
    df = pd.DataFrame(
        {
            "HasClaim": [1, 1, 1, 1, 0, 0],
            "ClaimAmountTotal": [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
        }
    )
    rate = compute_payout_given_claim_rate(df)
    assert np.isclose(rate, 0.75)


def test_validate_aggregate_loss_ratio():
    expected_loss = pd.Series([10.0, 20.0, 30.0])
    actual = pd.Series([5.0, 25.0, 30.0])
    result = validate_aggregate_loss(expected_loss, actual)
    assert np.isclose(result["total_predicted"], 60.0)
    assert np.isclose(result["total_actual"], 60.0)
    assert np.isclose(result["ratio_predicted_to_actual"], 1.0)
