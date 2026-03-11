"""
Agent 5: Modeling - Compare and select best model.
"""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def agent5_modeling(X, y):
    """Agent 5: Modeling."""
    print("=== Agent 5: Modeling ===")

    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        results[name] = {'mean_acc': scores.mean(), 'std_acc': scores.std()}
        print(f"{name}: Acc {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    best_model_name = max(results, key=lambda x: results[x]['mean_acc'])
    best_model = models[best_model_name]
    best_model.fit(X, y)

    return best_model, results

if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit
    from agent2_data_cleaning import agent2_data_cleaning
    from agent3_eda import agent3_eda
    from agent4_feature_engineering import agent4_feature_engineering
    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)
    train_clean, insights = agent3_eda(train_clean)
    X, y, X_test = agent4_feature_engineering(train_clean, test_clean, insights)
    best_model, model_results = agent5_modeling(X, y)