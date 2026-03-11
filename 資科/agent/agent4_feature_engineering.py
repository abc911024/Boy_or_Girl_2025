"""
Agent 4: Feature Engineering - Create features for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def agent4_feature_engineering(train_clean, test_clean, insights):
    """Agent 4: Feature Engineering."""
    print("=== Agent 4: Feature Engineering ===")

    cat_cols = ['star_sign', 'phone_os']
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_train = encoder.fit_transform(train_clean[cat_cols])
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols))
    train_clean = pd.concat([train_clean, encoded_train_df], axis=1)
    train_clean.drop(cat_cols, axis=1, inplace=True)

    encoded_test = encoder.transform(test_clean[cat_cols])
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols))
    test_clean = pd.concat([test_clean, encoded_test_df], axis=1)
    test_clean.drop(cat_cols, axis=1, inplace=True)

    # Text features
    for df in [train_clean, test_clean]:
        df['self_intro_len'] = df['self_intro'].str.len()
        df['self_intro_alpha_ratio'] = df['self_intro'].str.count(r'[a-zA-Z]') / (df['self_intro_len'] + 1)
        df['self_intro_digit_ratio'] = df['self_intro'].str.count(r'\d') / (df['self_intro_len'] + 1)
        df.drop('self_intro', axis=1, inplace=True)
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'self_intro_len', 'self_intro_alpha_ratio', 'self_intro_digit_ratio', 'bmi']
    scaler = StandardScaler()
    train_clean[num_cols] = scaler.fit_transform(train_clean[num_cols])
    test_clean[num_cols] = scaler.transform(test_clean[num_cols])

    X = train_clean[num_cols]
    y = train_clean['gender']
    X_test = test_clean[num_cols]

    print(f"Features: {X.shape[1]}")
    return X, y, X_test

if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit
    from agent2_data_cleaning import agent2_data_cleaning
    from agent3_eda import agent3_eda
    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)
    train_clean, insights = agent3_eda(train_clean)
    X, y, X_test = agent4_feature_engineering(train_clean, test_clean, insights)