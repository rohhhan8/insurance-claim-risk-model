"""Train/val/test split, done once and reused identically by frequency and severity.

Each row is already a unique IDpol (one row per policy-period after merge_clean),
so there's no repeated-entity structure to protect against via group-based
splitting -- a plain stratified random split is appropriate. Stratifying on
HasClaim guards against a validation/test fold with an unrepresentative claim
rate, which matters given the ~5% positive rate.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED_DIR, RANDOM_SEED, TEST_FRAC, VAL_FRAC

SPLIT_IDS_PATH = DATA_PROCESSED_DIR / "split_ids.json"


def split_data(
    df: pd.DataFrame, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split via two chained calls."""
    train_val, test = train_test_split(
        df, test_size=TEST_FRAC, stratify=df["HasClaim"], random_state=seed
    )
    val_frac_of_remainder = VAL_FRAC / (1 - TEST_FRAC)
    train, val = train_test_split(
        train_val,
        test_size=val_frac_of_remainder,
        stratify=train_val["HasClaim"],
        random_state=seed,
    )
    return train, val, test


def save_split_ids(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, path: Path = SPLIT_IDS_PATH
) -> None:
    """Persist IDpol membership per split so severity reuses the identical rows.

    The Step 5 expected-loss combination needs frequency and severity predictions on
    the same test policies, so both stages must load these same IDs rather than
    re-splitting independently.
    """
    ids = {
        "train": train["IDpol"].tolist(),
        "val": val["IDpol"].tolist(),
        "test": test["IDpol"].tolist(),
    }
    path.write_text(json.dumps(ids))


def load_split_ids(path: Path = SPLIT_IDS_PATH) -> dict:
    return json.loads(path.read_text())
