"""Shared constants so every pipeline stage agrees on the seed, splits, and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42

FREQ_DATA_ID = 41214  # freMTPL2freq on OpenML
SEV_DATA_ID = 41215  # freMTPL2sev on OpenML

# 70/15/15 train/val/test — the split.py module chains two calls to hit these fractions.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models_artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_FIGURES_DIR, REPORTS_METRICS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
