"""
Agent 5: Modeling - Improved model comparison and selection.
"""

import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def agent5_modeling(X, y):
    """Agent 5: Improved Modeling."""
    print("=== Agent 5: Modeling ===")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
    }

    results = {}

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        results[name] = {
            "mean_acc": float(scores.mean()),
            "std_acc": float(scores.std())
        }
        print(f"{name}: Acc {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    best_model_name = max(results, key=lambda k: results[k]["mean_acc"])
    best_model = models[best_model_name]
    best_model.fit(X, y)

    print(f"\nBest model: {best_model_name}")

    return best_model, results, best_model_name