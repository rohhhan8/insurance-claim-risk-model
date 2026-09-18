import pandas as pd

from src.data.merge_clean import aggregate_severity, clean, merge_freq_sev


def _base_freq_row(**overrides) -> dict:
    row = {
        "IDpol": 1.0,
        "ClaimNb": 0,
        "Exposure": 0.5,
        "VehGas": "'Regular'",
    }
    row.update(overrides)
    return row


def test_aggregate_severity_sums_multiple_claims():
    sev_df = pd.DataFrame(
        {"IDpol": [10, 10, 20], "ClaimAmount": [100.0, 250.0, 500.0]}
    )
    agg = aggregate_severity(sev_df)
    assert agg.set_index("IDpol").loc[10, "ClaimAmountTotal"] == 350.0
    assert agg.set_index("IDpol").loc[20, "ClaimAmountTotal"] == 500.0


def test_merge_freq_sev_fills_missing_with_zero():
    freq_df = pd.DataFrame([_base_freq_row(IDpol=1.0), _base_freq_row(IDpol=2.0)])
    sev_agg = pd.DataFrame({"IDpol": [1], "ClaimAmountTotal": [300.0]})
    merged = merge_freq_sev(freq_df, sev_agg)
    merged = merged.set_index("IDpol")
    assert merged.loc[1, "ClaimAmountTotal"] == 300.0
    assert merged.loc[2, "ClaimAmountTotal"] == 0.0


def test_exposure_capped_and_floored():
    df = pd.DataFrame(
        [
            _base_freq_row(IDpol=1.0, Exposure=1.5, ClaimAmountTotal=0.0),
            _base_freq_row(IDpol=2.0, Exposure=0.0001, ClaimAmountTotal=0.0),
        ]
    )
    cleaned, summary = clean(df)
    assert cleaned["Exposure"].max() <= 1.0
    assert cleaned["Exposure"].min() >= 1.0 / 365
    assert summary["exposure_capped_rows"] == 1
    assert summary["exposure_floored_rows"] == 1


def test_claimnb_capped():
    df = pd.DataFrame([_base_freq_row(IDpol=1.0, ClaimNb=20, ClaimAmountTotal=0.0)])
    cleaned, summary = clean(df)
    assert cleaned["ClaimNb"].iloc[0] == 4
    assert summary["claimnb_capped_rows"] == 1


def test_zero_claimnb_zeroes_claim_amount():
    df = pd.DataFrame([_base_freq_row(IDpol=1.0, ClaimNb=0, ClaimAmountTotal=500.0)])
    cleaned, summary = clean(df)
    assert cleaned["ClaimAmountTotal"].iloc[0] == 0.0
    assert summary["zero_claimnb_nonzero_amount_rows"] == 1


def test_hasclaim_derived_correctly():
    df = pd.DataFrame(
        [
            _base_freq_row(IDpol=1.0, ClaimNb=0, ClaimAmountTotal=0.0),
            _base_freq_row(IDpol=2.0, ClaimNb=2, ClaimAmountTotal=400.0),
        ]
    )
    cleaned, _ = clean(df)
    assert cleaned.set_index("IDpol").loc[1.0, "HasClaim"] == 0
    assert cleaned.set_index("IDpol").loc[2.0, "HasClaim"] == 1


def test_vehgas_quotes_stripped():
    df = pd.DataFrame(
        [_base_freq_row(IDpol=1.0, VehGas="'Diesel'", ClaimAmountTotal=0.0)]
    )
    cleaned, _ = clean(df)
    assert cleaned["VehGas"].iloc[0] == "Diesel"
