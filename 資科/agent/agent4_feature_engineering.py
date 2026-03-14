"""
Agent 4: Feature Engineering - Improved version with
numeric features, derived features, one-hot categories, and TF-IDF text features.
"""

import pandas as pd
import numpy as np

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric features."""
    df = df.copy()

    df["self_intro_len"] = df["self_intro"].astype(str).str.len()
    df["self_intro_word_count"] = df["self_intro"].astype(str).str.split().str.len()
    df["self_intro_alpha_ratio"] = df["self_intro"].astype(str).str.count(r"[a-zA-Z]") / (df["self_intro_len"] + 1)
    df["self_intro_digit_ratio"] = df["self_intro"].astype(str).str.count(r"\d") / (df["self_intro_len"] + 1)
    df["self_intro_upper_ratio"] = df["self_intro"].astype(str).str.count(r"[A-Z]") / (df["self_intro_len"] + 1)
    df["self_intro_punct_ratio"] = df["self_intro"].astype(str).str.count(r"[^\w\s]") / (df["self_intro_len"] + 1)

    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["social_activity"] = df["fb_friends"] + df["yt"]
    df["brain_sleep_interaction"] = df["iq"] * df["sleepiness"]
    df["height_weight_ratio"] = df["height"] / (df["weight"] + 1)

    return df


def agent4_feature_engineering(train_clean, test_clean, insights):
    """Agent 4: Improved Feature Engineering."""
    print("=== Agent 4: Feature Engineering ===")

    train_clean = train_clean.copy()
    test_clean = test_clean.copy()

    # Add derived features
    train_clean = add_derived_features(train_clean)
    test_clean = add_derived_features(test_clean)

    # Numeric features
    num_cols = [
        "height", "weight", "sleepiness", "iq", "fb_friends", "yt",
        "self_intro_len", "self_intro_word_count",
        "self_intro_alpha_ratio", "self_intro_digit_ratio",
        "self_intro_upper_ratio", "self_intro_punct_ratio",
        "bmi", "social_activity", "brain_sleep_interaction", "height_weight_ratio"
    ]

    # Categorical features
    cat_cols = ["star_sign", "phone_os"]

    # Standardize numeric features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_clean[num_cols])
    X_test_num = scaler.transform(test_clean[num_cols])

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_train_cat = encoder.fit_transform(train_clean[cat_cols])
    X_test_cat = encoder.transform(test_clean[cat_cols])

    # TF-IDF for self_intro
    vectorizer = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_text = vectorizer.fit_transform(train_clean["self_intro"].fillna("").astype(str))
    X_test_text = vectorizer.transform(test_clean["self_intro"].fillna("").astype(str))

    # Combine all features
    X_train = hstack([
        csr_matrix(X_train_num),
        X_train_cat,
        X_train_text
    ]).tocsr()

    X_test = hstack([
        csr_matrix(X_test_num),
        X_test_cat,
        X_test_text
    ]).tocsr()

    y = train_clean["gender"].astype(int)

    feature_info = {
        "numeric_features": num_cols,
        "categorical_features": encoder.get_feature_names_out(cat_cols).tolist(),
        "text_feature_count": X_train_text.shape[1],
        "total_feature_count": X_train.shape[1]
    }

    print(f"Numeric feature count: {len(num_cols)}")
    print(f"Categorical one-hot feature count: {X_train_cat.shape[1]}")
    print(f"TF-IDF feature count: {X_train_text.shape[1]}")
    print(f"Total feature count: {X_train.shape[1]}")

    return X_train, y, X_test, feature_info


if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit
    from agent2_data_cleaning import agent2_data_cleaning
    from agent3_eda import agent3_eda

    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)
    train_clean, insights = agent3_eda(train_clean)
    X, y, X_test, feature_info = agent4_feature_engineering(train_clean, test_clean, insights)