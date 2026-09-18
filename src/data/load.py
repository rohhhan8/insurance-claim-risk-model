"""Fetch the French MTPL frequency and severity datasets from OpenML, with a local cache.

OpenML fetches are slow (network round-trip + server-side parsing) and occasionally
flaky, so every fetch is cached to parquet on first successful call and read from
disk afterward.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from src.config import DATA_RAW_DIR, FREQ_DATA_ID, SEV_DATA_ID

FREQ_CACHE_PATH = DATA_RAW_DIR / "freq_raw.parquet"
SEV_CACHE_PATH = DATA_RAW_DIR / "sev_raw.parquet"


def fetch_frequency_data(cache_path: Path = FREQ_CACHE_PATH) -> pd.DataFrame:
    """Load freMTPL2freq (one row per policy-period), including IDpol and ClaimNb.

    Returned as the full `.frame` rather than splitting into `.data`/`.target` —
    OpenML's target assignment for this dataset isn't consistent across sklearn
    versions, and we need every column (including IDpol) regardless.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    bunch = fetch_openml(data_id=FREQ_DATA_ID, as_frame=True, parser="auto")
    df = bunch.frame
    df.to_parquet(cache_path)
    return df


def fetch_severity_data(cache_path: Path = SEV_CACHE_PATH) -> pd.DataFrame:
    """Load freMTPL2sev: one row per individual claim (IDpol, ClaimAmount)."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    bunch = fetch_openml(data_id=SEV_DATA_ID, as_frame=True, parser="auto")
    df = bunch.frame
    df.to_parquet(cache_path)
    return df
