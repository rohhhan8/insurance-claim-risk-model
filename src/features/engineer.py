"""Categorical encoding and derived features for the frequency/severity models.

The dataset is cross-sectional (one row per policy-period, no repeated IDpol over
time), so there's no way to build genuine "claims in the last N years" history
features. BonusMalus (France's no-claims bonus/malus score) is the only field that
embeds real driving history, which is why it anchors the one history-proxy feature
below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

ONE_HOT_CARDINALITY_THRESHOLD = 25
CATEGORICAL_COLUMNS = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC_COLUMNS = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add three non-obvious ratio/proxy features on top of the raw columns.

    Each is justified individually because none is a standard textbook feature for
    this dataset — they're constructed to compensate for what's missing (no panel
    history) or to linearize what's badly skewed (Density).
    """
    df = df.copy()

    # Density spans ~1 to ~27,000 and is heavily right-skewed; log1p keeps a handful
    # of extreme-density regions from dominating tree splits on raw magnitude alone.
    df["log_density"] = np.log1p(df["Density"])

    # Proxies "old car relative to driver's age" as a distinct risk signal that
    # neither VehAge nor DrivAge alone conveys (a young driver in a very old car vs.
    # an experienced driver in a new one are different risk profiles). "+1" is a
    # defensive no-op here since DrivAge's minimum in this data is 18.
    df["vehage_drivage_ratio"] = df["VehAge"] / (df["DrivAge"] + 1)

    # BonusMalus is the dataset's only embedded proxy for driving history. Dividing
    # by an approximate years-licensed-since-18 flags drivers whose malus score is
    # unusually high/low GIVEN plausible years of driving experience, rather than
    # raw magnitude alone (e.g. a 20-year-old at malus=70 is a different risk signal
    # than a 50-year-old at malus=70). This assumes a minimum licensing age of 18,
    # which the data doesn't state directly — an approximation, not a measured value.
    years_licensed_proxy = (df["DrivAge"] - 17).clip(lower=1)
    df["bonus_malus_per_year_licensed"] = df["BonusMalus"] / years_licensed_proxy

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    fit_encoders: bool,
    encoders: dict | None = None,
    target: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Encode categoricals and assemble the final feature matrix.

    `fit_encoders=True` on train fits new encoder objects; `fit_encoders=False` on
    val/test reuses the encoders returned from the train call. Fitting separately per
    split (rather than fitting once on the full dataset) is what prevents target/
    distribution leakage from val/test into the training encoders.
    """
    df = add_derived_features(df)
    encoders = {} if encoders is None else dict(encoders)

    numeric_and_derived = NUMERIC_COLUMNS + [
        "log_density",
        "vehage_drivage_ratio",
        "bonus_malus_per_year_licensed",
    ]
    pieces = [df[numeric_and_derived].reset_index(drop=True)]

    for col in CATEGORICAL_COLUMNS:
        cardinality = df[col].nunique()
        # Threshold-based branch kept generic (not hardcoded per column) even though
        # every column in this dataset falls under 25 unique values in practice —
        # a higher-cardinality categorical would automatically route to target
        # encoding instead of blowing up into 100+ one-hot columns.
        if cardinality <= ONE_HOT_CARDINALITY_THRESHOLD:
            key = f"onehot_{col}"
            if fit_encoders:
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoded = enc.fit_transform(df[[col]])
                encoders[key] = enc
            else:
                enc = encoders[key]
                encoded = enc.transform(df[[col]])
            col_names = [f"{col}_{cat}" for cat in enc.categories_[0]]
            pieces.append(pd.DataFrame(encoded, columns=col_names))
        else:
            key = f"target_{col}"
            if fit_encoders:
                enc = TargetEncoder(random_state=0)
                encoded = enc.fit_transform(df[[col]], target)
                encoders[key] = enc
            else:
                enc = encoders[key]
                encoded = enc.transform(df[[col]])
            pieces.append(pd.DataFrame(encoded, columns=[f"{col}_target_enc"]))

    feature_matrix = pd.concat(pieces, axis=1)
    return feature_matrix, encoders
