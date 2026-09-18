"""Merge freMTPL2freq with freMTPL2sev and apply data-quality cleaning rules.

Every cleaning decision here is a data-quality correction applied uniformly across
the whole population, not a label-informed transform — so it's safe to run before
the train/val/test split (unlike class-imbalance handling, which must happen after
the split using train-only statistics).
"""

import pandas as pd

from src.config import DATA_PROCESSED_DIR

CLAIMNB_CAP = 4  # a handful of rows report 9-16 claims against a single policy-year,
# which is not plausible for third-party liability exposure and skews scale_pos_weight
# and any rate-based features. 4 matches the cap used in sklearn's own worked example
# on this dataset (the Poisson/Tweedie GLM tutorial), so results stay comparable to
# the standard reference treatment of this data.
EXPOSURE_MIN = 1.0 / 365  # ~1 day; guards against divide-by-zero in exposure-normalized
# features for the few rows with near-zero exposure but a recorded claim.
EXPOSURE_MAX = 1.0  # a policy-year cannot exceed one year of exposure; values above 1
# (observed up to 2.01 in the raw data) are a data entry artifact, not a real quantity.


def aggregate_severity(sev_df: pd.DataFrame) -> pd.DataFrame:
    """Sum claim amounts per IDpol.

    A policy can have multiple claim rows (observed up to 5 in this data). Summing,
    rather than taking the max or first row, is what makes the result the correct
    total-loss counterpart to ClaimNb, which already counts claims per policy.
    """
    sev_agg = sev_df.groupby("IDpol")["ClaimAmount"].sum().rename("ClaimAmountTotal")
    return sev_agg.reset_index()


def merge_freq_sev(freq_df: pd.DataFrame, sev_agg: pd.DataFrame) -> pd.DataFrame:
    """Left-join on IDpol; policies with no matching claim get ClaimAmountTotal=0."""
    freq_df = freq_df.copy()
    sev_agg = sev_agg.copy()
    # freq's IDpol comes back as float64 from OpenML, sev's as int64 — cast both to
    # int64 before joining or every match silently fails on dtype mismatch.
    freq_df["IDpol"] = freq_df["IDpol"].astype("int64")
    sev_agg["IDpol"] = sev_agg["IDpol"].astype("int64")

    merged = freq_df.merge(sev_agg, on="IDpol", how="left")
    merged["ClaimAmountTotal"] = merged["ClaimAmountTotal"].fillna(0.0)
    return merged


def clean(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply cleaning rules and return the cleaned frame plus a summary of what changed.

    The summary dict is printed by the caller / surfaced in the EDA notebook so the
    magnitude of each correction is visible rather than silently baked into the data.
    """
    df = merged.copy()
    summary = {}

    # VehGas values arrive wrapped in literal quote characters (e.g. "'Diesel'") from
    # OpenML's ARFF parsing — strip them so the category doesn't get one-hot encoded
    # as a distinct, oddly-named level from a clean "Diesel".
    df["VehGas"] = df["VehGas"].str.strip("'")

    exposure_capped = (df["Exposure"] > EXPOSURE_MAX).sum()
    exposure_floored = (df["Exposure"] < EXPOSURE_MIN).sum()
    df["Exposure"] = df["Exposure"].clip(lower=EXPOSURE_MIN, upper=EXPOSURE_MAX)
    summary["exposure_capped_rows"] = int(exposure_capped)
    summary["exposure_floored_rows"] = int(exposure_floored)

    claimnb_capped = (df["ClaimNb"] > CLAIMNB_CAP).sum()
    df["ClaimNb"] = df["ClaimNb"].clip(upper=CLAIMNB_CAP)
    summary["claimnb_capped_rows"] = int(claimnb_capped)

    # ClaimNb==0 but a nonzero recorded amount is treated as noise: ClaimNb is the
    # more reliable field (it's the frequency table's own count), so we trust it and
    # zero out the inconsistent amount rather than inventing a phantom claim.
    inconsistent_zero_claims = ((df["ClaimNb"] == 0) & (df["ClaimAmountTotal"] > 0)).sum()
    df.loc[df["ClaimNb"] == 0, "ClaimAmountTotal"] = 0.0
    summary["zero_claimnb_nonzero_amount_rows"] = int(inconsistent_zero_claims)

    # ClaimNb>0 but ClaimAmountTotal==0 affects ~27% of claiming policies here (9,116
    # of 34,060) — too large a fraction to be purely "claim reported, no payout."
    # This is a documented quirk of this exact dataset: freMTPL2freq and freMTPL2sev
    # were extracted from the source system somewhat independently, so a nontrivial
    # number of policies counted as claims in the frequency table have no matching
    # row in the severity table at all. We leave ClaimNb as the ground truth for
    # frequency (it's what the frequency model targets) and leave ClaimAmountTotal=0
    # for these rows rather than imputing a fabricated severity — the severity model
    # is trained only on rows with a real recorded ClaimAmountTotal>0 (see
    # src/models/severity.py), so these rows simply don't contribute to severity
    # training, which is the correct treatment given we can't know their true payout.
    zero_payment_claims = ((df["ClaimNb"] > 0) & (df["ClaimAmountTotal"] == 0)).sum()
    summary["nonzero_claimnb_zero_amount_rows"] = int(zero_payment_claims)

    # Large severity values are NOT capped or dropped: the right tail is exactly what
    # an insurer needs to price for, and the severity model's evaluation separately
    # reports error on the top 10% of claims (see src/models/severity.py) instead of
    # hiding that behavior by trimming it here.
    top_01_pct_threshold = df.loc[df["ClaimAmountTotal"] > 0, "ClaimAmountTotal"].quantile(0.999)
    summary["top_0.1pct_severity_threshold"] = float(top_01_pct_threshold)

    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)

    summary["n_rows"] = int(len(df))
    summary["claim_rate"] = float(df["HasClaim"].mean())

    return df, summary


def build_merged_clean_dataset(
    freq_df: pd.DataFrame, sev_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """End-to-end: aggregate severity, merge onto frequency, apply cleaning rules."""
    sev_agg = aggregate_severity(sev_df)
    merged = merge_freq_sev(freq_df, sev_agg)
    cleaned, summary = clean(merged)
    return cleaned, summary


def save_processed(df: pd.DataFrame, path=DATA_PROCESSED_DIR / "merged_clean.parquet") -> None:
    df.to_parquet(path)
