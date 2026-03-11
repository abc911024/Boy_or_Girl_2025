"""
Agent 2: Data Cleaning - Preprocessing train and test data.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def clean_df(df, is_train=True):
    """Clean data consistently."""
    # Fix yt
    df['yt'] = pd.to_numeric(df['yt'], errors='coerce')
    # Outlier handling with IQR
    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']
    for col in num_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.where((df[col] < lower) | (df[col] > upper), np.nan, df[col])
    # Impute
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[['star_sign', 'phone_os']] = cat_imputer.fit_transform(df[['star_sign', 'phone_os']])
    df['self_intro'] = df['self_intro'].fillna('')
    # Handle abnormal
    df['phone_os'] = df['phone_os'].replace('JohnCena', 'Android')
    # Dtypes
    df['id'] = df['id'].astype(int)
    if is_train:
        df['gender'] = df['gender'].astype(int)
    df[['star_sign', 'phone_os', 'self_intro']] = df[['star_sign', 'phone_os', 'self_intro']].astype(str)
    df[num_cols] = df[num_cols].astype(float)
    return df

def agent2_data_cleaning(train, test):
    """Agent 2: Data Cleaning."""
    print("=== Agent 2: Data Cleaning ===")
    train_clean = clean_df(train, True)
    test_clean = clean_df(test, False)
    print(f"Train clean missing: {train_clean.isnull().sum().sum()}")
    print(f"Test clean missing: {test_clean.isnull().sum().sum()}")
    return train_clean, test_clean

if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit
    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)