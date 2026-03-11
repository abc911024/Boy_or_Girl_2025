"""
Agent 3: EDA - Exploratory Data Analysis.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

def agent3_eda(train_clean):
    """Agent 3: EDA."""
    print("=== Agent 3: EDA ===")
    target = 'gender'
    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']
    cat_cols = ['star_sign', 'phone_os']

    # Numerical vs gender
    for col in num_cols:
        group1 = train_clean[train_clean[target] == 1][col]
        group2 = train_clean[train_clean[target] == 2][col]
        t_stat, p = ttest_ind(group1, group2, nan_policy='omit')
        print(f"{col}: p={p:.4f}")

    # Categorical vs gender
    for col in cat_cols:
        contingency = pd.crosstab(train_clean[col], train_clean[target])
        chi2, p, _, _ = chi2_contingency(contingency)
        print(f"{col}: chi2={chi2:.2f}, p={p:.4f}")

    # Text length
    train_clean['self_intro_len'] = train_clean['self_intro'].str.len()
    len1 = train_clean[train_clean[target] == 1]['self_intro_len']
    len2 = train_clean[train_clean[target] == 2]['self_intro_len']
    t_len, p_len = ttest_ind(len1, len2, nan_policy='omit')
    print(f"Self_intro_len: p={p_len:.4f}")

    # BMI
    train_clean['bmi'] = train_clean['weight'] / (train_clean['height'] / 100) ** 2
    bmi1 = train_clean[train_clean[target] == 1]['bmi']
    bmi2 = train_clean[train_clean[target] == 2]['bmi']
    t_bmi, p_bmi = ttest_ind(bmi1, bmi2, nan_policy='omit')
    print(f"BMI: p={p_bmi:.4f}")

    insights = {
        'significant_num': [col for col in num_cols if ttest_ind(train_clean[train_clean[target] == 1][col], train_clean[train_clean[target] == 2][col], nan_policy='omit')[1] < 0.05],
        'significant_cat': [col for col in cat_cols if chi2_contingency(pd.crosstab(train_clean[col], train_clean[target]))[1] < 0.05],
        'text_features': ['self_intro_len', 'bmi']
    }
    return train_clean, insights

if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit
    from agent2_data_cleaning import agent2_data_cleaning
    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)
    train_clean, insights = agent3_eda(train_clean)