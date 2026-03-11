"""
Agent 1: Data Audit - Quality check for train and test data.
"""

import pandas as pd
import numpy as np

def load_data():
    """Load train, test, and sample data."""
    train = pd.read_csv('boy or girl 2025 train_missingValue.csv')
    test = pd.read_csv('boy or girl 2025 test no ans_missingValue.csv')
    sample = pd.read_csv('Boy_or_girl_test_sandbox_sample_submission.csv')
    return train, test, sample

def agent1_data_audit(train, test):
    """Agent 1: Data Audit - Quality check."""
    print("=== Agent 1: Data Audit ===")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Train dtypes:\n{train.dtypes}")
    print(f"Test dtypes:\n{test.dtypes}")
    print(f"Train missing:\n{train.isnull().sum()}")
    print(f"Test missing:\n{test.isnull().sum()}")
    print(f"Gender distribution:\n{train['gender'].value_counts()}")
    print(f"Star_sign unique: {sorted(train['star_sign'].dropna().unique())}")
    print(f"Phone_os unique: {sorted(train['phone_os'].dropna().unique())}")
    print(f"YT sample: {train['yt'].dropna().unique()[:10]}")
    print(f"Self_intro sample: {train['self_intro'].dropna().head(5).tolist()}")
    print(f"Numerical describe:\n{train[['height', 'weight', 'sleepiness', 'iq', 'fb_friends']].describe()}")
    print(f"Duplicates: {train.duplicated().sum()}, ID duplicates: {train['id'].duplicated().sum()}")
    return train, test

if __name__ == "__main__":
    train, test, sample = load_data()
    agent1_data_audit(train, test)