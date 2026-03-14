"""
Agent 2: Data Cleaning - Improved preprocessing
with model-based categorical imputation.
"""

import re
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def create_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight text features from self_intro."""
    df = df.copy()
    text = df["self_intro"].fillna("").astype(str)

    df["self_intro_len"] = text.str.len()
    df["self_intro_word_count"] = text.str.split().str.len()
    df["self_intro_alpha_ratio"] = text.str.count(r"[a-zA-Z]") / (df["self_intro_len"] + 1)
    df["self_intro_digit_ratio"] = text.str.count(r"\d") / (df["self_intro_len"] + 1)
    df["self_intro_upper_ratio"] = text.str.count(r"[A-Z]") / (df["self_intro_len"] + 1)
    df["self_intro_punct_ratio"] = text.str.count(r"[^\w\s]") / (df["self_intro_len"] + 1)

    return df


def basic_clean_df(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Basic cleaning before model-based categorical imputation.
    - Fix yt to numeric
    - Replace abnormal phone_os value JohnCena -> Unknown
    - Normalize blanks to NaN
    - Fill self_intro with empty string
    - Create text features
    - Normalize dtypes
    """
    df = df.copy()

    # Normalize blank-like strings
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(
                ["", " ", "nan", "None", "null", "NULL", "NaN"],
                np.nan
            )

    # Fix yt
    if "yt" in df.columns:
        df["yt"] = pd.to_numeric(df["yt"], errors="coerce")

    # Replace abnormal category
    if "phone_os" in df.columns:
        df["phone_os"] = df["phone_os"].replace("JohnCena", "Unknown")

    # Ensure numeric columns
    num_cols = ["height", "weight", "sleepiness", "iq", "fb_friends", "yt"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill self_intro first for feature extraction
    if "self_intro" in df.columns:
        df["self_intro"] = df["self_intro"].fillna("").astype(str)
    else:
        df["self_intro"] = ""

    # Create text features
    df = create_text_features(df)

    # Normalize dtypes
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype(int)

    if is_train and "gender" in df.columns:
        df["gender"] = pd.to_numeric(df["gender"], errors="coerce").astype("Int64")

    # Make sure categorical cols remain object
    for col in ["star_sign", "phone_os"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # Numeric feature dtypes
    feature_num_cols = num_cols + [
        "self_intro_len",
        "self_intro_word_count",
        "self_intro_alpha_ratio",
        "self_intro_digit_ratio",
        "self_intro_upper_ratio",
        "self_intro_punct_ratio"
    ]
    existing_feature_num_cols = [c for c in feature_num_cols if c in df.columns]
    df[existing_feature_num_cols] = df[existing_feature_num_cols].astype(float)

    return df


def handle_numeric_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str]):
    """
    Fit IQR thresholds on train only, then apply to both train and test.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in num_cols:
        if col not in train_df.columns:
            continue

        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        train_df.loc[(train_df[col] < lower) | (train_df[col] > upper), col] = np.nan
        if col in test_df.columns:
            test_df.loc[(test_df[col] < lower) | (test_df[col] > upper), col] = np.nan

    return train_df, test_df


def impute_numeric(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str]):
    """
    Fit numeric imputer on train only.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    existing_cols = [c for c in num_cols if c in train_df.columns]
    imputer = SimpleImputer(strategy="median")

    train_df[existing_cols] = imputer.fit_transform(train_df[existing_cols])
    test_df[existing_cols] = imputer.transform(test_df[existing_cols])

    return train_df, test_df


def model_impute_categorical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    min_samples: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute a categorical column using a classification model trained
    on non-missing rows from the training data only.

    Fallback:
    - if too few samples
    - if only one class exists
    then fill missing with mode or 'Unknown'
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    known_train = train_df[train_df[target_col].notna()].copy()

    fallback_value = "Unknown"
    if len(known_train) > 0:
        fallback_value = known_train[target_col].mode(dropna=True)
        fallback_value = fallback_value.iloc[0] if len(fallback_value) > 0 else "Unknown"

    if len(known_train) < min_samples or known_train[target_col].nunique() < 2:
        train_df[target_col] = train_df[target_col].fillna(fallback_value)
        test_df[target_col] = test_df[target_col].fillna(fallback_value)
        return train_df, test_df

    X_train_known = known_train[feature_cols]
    y_train_known = known_train[target_col].astype(str)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_known, y_train_known)

    train_missing_mask = train_df[target_col].isna()
    if train_missing_mask.any():
        X_train_missing = train_df.loc[train_missing_mask, feature_cols]
        train_df.loc[train_missing_mask, target_col] = clf.predict(X_train_missing)

    test_missing_mask = test_df[target_col].isna()
    if test_missing_mask.any():
        X_test_missing = test_df.loc[test_missing_mask, feature_cols]
        test_df.loc[test_missing_mask, target_col] = clf.predict(X_test_missing)

    train_df[target_col] = train_df[target_col].fillna(fallback_value)
    test_df[target_col] = test_df[target_col].fillna(fallback_value)

    return train_df, test_df


def agent2_data_cleaning(train: pd.DataFrame, test: pd.DataFrame):
    """Agent 2: Improved Data Cleaning."""
    print("=== Agent 2: Data Cleaning ===")

    train_clean = basic_clean_df(train, is_train=True)
    test_clean = basic_clean_df(test, is_train=False)

    num_cols = ["height", "weight", "sleepiness", "iq", "fb_friends", "yt"]
    text_cols = [
        "self_intro_len",
        "self_intro_word_count",
        "self_intro_alpha_ratio",
        "self_intro_digit_ratio",
        "self_intro_upper_ratio",
        "self_intro_punct_ratio"
    ]

    # Outlier handling based on train only
    train_clean, test_clean = handle_numeric_outliers(train_clean, test_clean, num_cols)

    # Numeric imputation based on train only
    train_clean, test_clean = impute_numeric(train_clean, test_clean, num_cols + text_cols)

    # Features for categorical imputation
    feature_cols = num_cols + text_cols

    # Model-based imputation
    for target_col in ["star_sign", "phone_os"]:
        train_clean, test_clean = model_impute_categorical(
            train_clean,
            test_clean,
            target_col=target_col,
            feature_cols=feature_cols
        )

    # Final dtype normalization
    for df in [train_clean, test_clean]:
        df["star_sign"] = df["star_sign"].astype(str)
        df["phone_os"] = df["phone_os"].astype(str)
        df["self_intro"] = df["self_intro"].astype(str)

    print(f"Train clean missing: {train_clean.isnull().sum().sum()}")
    print(f"Test clean missing: {test_clean.isnull().sum().sum()}")

    return train_clean, test_clean


if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit

    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)

    print("\n[Check] star_sign missing:", train_clean["star_sign"].isnull().sum(), test_clean["star_sign"].isnull().sum())
    print("[Check] phone_os missing:", train_clean["phone_os"].isnull().sum(), test_clean["phone_os"].isnull().sum())
    print("[Check] phone_os unique:", sorted(train_clean["phone_os"].unique()))