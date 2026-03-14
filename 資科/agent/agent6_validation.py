"""
Agent 6: Validation - Improved validation with multiple metrics.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def agent6_validation(best_model, X, y):
    """Agent 6: Improved Validation."""
    print("=== Agent 6: Validation ===")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, precs, recs = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = clone(best_model)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, pos_label=1)
        prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

        accs.append(acc)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

        print(
            f"Fold {fold}: "
            f"Acc={acc:.4f}, "
            f"F1={f1:.4f}, "
            f"Precision={prec:.4f}, "
            f"Recall={rec:.4f}"
        )

    print("\n[Validation Summary]")
    print(f"Accuracy : {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    print(f"F1       : {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
    print(f"Precision: {np.mean(precs):.4f} (+/- {np.std(precs):.4f})")
    print(f"Recall   : {np.mean(recs):.4f} (+/- {np.std(recs):.4f})")

    return {
        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "precision_mean": np.mean(precs),
        "precision_std": np.std(precs),
        "recall_mean": np.mean(recs),
        "recall_std": np.std(recs)
    }


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
    X, y, X_test, feature_info = agent4_feature_engineering(train_clean, test_clean, insights)
    best_model, model_results, best_model_name = agent5_modeling(X, y)
    validation_results = agent6_validation(best_model, X, y)