"""
Agent 3: EDA - Improved Exploratory Data Analysis.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency


def cohen_d(x1, x2):
    """Simple Cohen's d effect size."""
    x1 = np.array(x1.dropna(), dtype=float)
    x2 = np.array(x2.dropna(), dtype=float)

    if len(x1) < 2 or len(x2) < 2:
        return np.nan

    pooled_std = np.sqrt(((x1.std(ddof=1) ** 2) + (x2.std(ddof=1) ** 2)) / 2)
    if pooled_std == 0:
        return 0.0
    return (x1.mean() - x2.mean()) / pooled_std


def agent3_eda(train_clean: pd.DataFrame):
    """Agent 3: Improved EDA."""
    print("=== Agent 3: EDA ===")

    train_clean = train_clean.copy()
    target = "gender"

    num_cols = ["height", "weight", "sleepiness", "iq", "fb_friends", "yt"]
    cat_cols = ["star_sign", "phone_os"]

    # Derived features for analysis
    train_clean["self_intro_len"] = train_clean["self_intro"].astype(str).str.len()
    train_clean["bmi"] = train_clean["weight"] / ((train_clean["height"] / 100) ** 2)
    train_clean["social_activity"] = train_clean["fb_friends"] + train_clean["yt"]
    train_clean["brain_sleep_interaction"] = train_clean["iq"] * train_clean["sleepiness"]

    extra_num_cols = ["self_intro_len", "bmi", "social_activity", "brain_sleep_interaction"]
    all_num_cols = num_cols + extra_num_cols

    numeric_results = []
    categorical_results = []

    group1_df = train_clean[train_clean[target] == 1]
    group2_df = train_clean[train_clean[target] == 2]

    print("\n[Numeric Features]")
    for col in all_num_cols:
        group1 = group1_df[col]
        group2 = group2_df[col]

        t_stat, p = ttest_ind(group1, group2, nan_policy="omit")
        effect = cohen_d(group1, group2)

        result = {
            "feature": col,
            "mean_gender1": group1.mean(),
            "mean_gender2": group2.mean(),
            "p_value": p,
            "cohen_d": effect
        }
        numeric_results.append(result)

        print(
            f"{col}: "
            f"mean(g1)={group1.mean():.3f}, "
            f"mean(g2)={group2.mean():.3f}, "
            f"p={p:.4f}, d={effect:.3f}"
        )

    print("\n[Categorical Features]")
    for col in cat_cols:
        contingency = pd.crosstab(train_clean[col], train_clean[target])
        chi2, p, _, _ = chi2_contingency(contingency)

        result = {
            "feature": col,
            "chi2": chi2,
            "p_value": p,
            "n_categories": contingency.shape[0]
        }
        categorical_results.append(result)

        print(f"{col}: chi2={chi2:.3f}, p={p:.4f}, categories={contingency.shape[0]}")

    numeric_results_df = pd.DataFrame(numeric_results).sort_values(by="p_value")
    categorical_results_df = pd.DataFrame(categorical_results).sort_values(by="p_value")

    insights = {
        "significant_num": numeric_results_df.loc[numeric_results_df["p_value"] < 0.05, "feature"].tolist(),
        "significant_cat": categorical_results_df.loc[categorical_results_df["p_value"] < 0.05, "feature"].tolist(),
        "recommended_numeric_features": numeric_results_df["feature"].tolist(),
        "recommended_categorical_features": categorical_results_df["feature"].tolist(),
        "numeric_summary": numeric_results_df,
        "categorical_summary": categorical_results_df
    }

    print("\n[EDA Summary]")
    print("Significant numeric features:", insights["significant_num"])
    print("Significant categorical features:", insights["significant_cat"])

    return train_clean, insights


if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit
    from agent2_data_cleaning import agent2_data_cleaning

    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)
    train_clean, insights = agent3_eda(train_clean)