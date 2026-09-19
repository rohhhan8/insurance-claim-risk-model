import numpy as np
import pandas as pd

from src.features.engineer import add_derived_features, build_feature_matrix


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Exposure": [0.5, 1.0, 0.3],
            "VehPower": [5, 7, 4],
            "VehAge": [3, 0, 10],
            "DrivAge": [30, 18, 60],
            "BonusMalus": [50, 100, 60],
            "Density": [100, 0, 27000],
            "Area": ["A", "B", "A"],
            "VehBrand": ["B1", "B2", "B1"],
            "VehGas": ["Diesel", "Regular", "Diesel"],
            "Region": ["R11", "R22", "R11"],
        }
    )


def test_log_density_transform_no_nan_on_zero():
    df = add_derived_features(_sample_df())
    assert np.isclose(df["log_density"].iloc[1], np.log1p(0))
    assert df["log_density"].notna().all()


def test_vehage_drivage_ratio_no_div_by_zero():
    df = add_derived_features(_sample_df())
    expected = 0 / (18 + 1)
    assert np.isclose(df["vehage_drivage_ratio"].iloc[1], expected)
    assert df["vehage_drivage_ratio"].notna().all()


def test_bonus_malus_per_year_licensed_floors_at_one_year():
    df = add_derived_features(_sample_df())
    # DrivAge=18 -> years_licensed_proxy would be 1 (clipped from 18-17=1)
    assert np.isclose(df["bonus_malus_per_year_licensed"].iloc[1], 100 / 1)


def test_build_feature_matrix_no_leakage_on_unseen_category():
    train = _sample_df()
    test = _sample_df()
    test.loc[0, "VehBrand"] = "B_UNSEEN"

    X_train, encoders = build_feature_matrix(train, fit_encoders=True)
    X_test, _ = build_feature_matrix(test, fit_encoders=False, encoders=encoders)

    assert set(X_test.columns) == set(X_train.columns)
    unseen_row_onehot_cols = [c for c in X_test.columns if c.startswith("VehBrand_")]
    assert X_test.loc[0, unseen_row_onehot_cols].sum() == 0
