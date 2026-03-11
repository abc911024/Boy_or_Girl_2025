"""
Agent 6: Validation - Validate model performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def agent6_validation(best_model, X, y):
    """Agent 6: Validation."""
    print("=== Agent 6: Validation ===")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, precs, recs = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_val)
        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, pos_label=1))
        precs.append(precision_score(y_val, y_pred, pos_label=1))
        recs.append(recall_score(y_val, y_pred, pos_label=1))

    print(f"Accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    print(f"F1: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
    print(f"Precision: {np.mean(precs):.4f} (+/- {np.std(precs):.4f})")
    print(f"Recall: {np.mean(recs):.4f} (+/- {np.std(recs):.4f})")

    return best_model

if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit
    from agent2_data_cleaning import agent2_data_cleaning
    from agent3_eda import agent3_eda
    from agent4_feature_engineering import agent4_feature_engineering
    from agent5_modeling import agent5_modeling
    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)
    train_clean, insights = agent3_eda(train_clean)
    X, y, X_test = agent4_feature_engineering(train_clean, test_clean, insights)
    best_model, model_results = agent5_modeling(X, y)
    best_model = agent6_validation(best_model, X, y)