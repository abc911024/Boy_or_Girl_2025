"""
Agent 7: Submission - Generate submission file.
"""

import pandas as pd

def agent7_submission(best_model, X_test, test_clean, sample):
    """Agent 7: Submission."""
    print("=== Agent 7: Submission ===")

    y_pred = best_model.predict(X_test)
    submission = pd.DataFrame({'id': test_clean['id'], 'gender': y_pred})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
    print(f"Sample submission shape: {sample.shape}")
    print(f"Our submission shape: {submission.shape}")
    print(f"Gender distribution: {submission['gender'].value_counts()}")

    return submission

if __name__ == "__main__":
    from agent1_data_audit import load_data, agent1_data_audit
    from agent2_data_cleaning import agent2_data_cleaning
    from agent3_eda import agent3_eda
    from agent4_feature_engineering import agent4_feature_engineering
    from agent5_modeling import agent5_modeling
    from agent6_validation import agent6_validation
    train, test, sample = load_data()
    train, test = agent1_data_audit(train, test)
    train_clean, test_clean = agent2_data_cleaning(train, test)
    train_clean, insights = agent3_eda(train_clean)
    X, y, X_test = agent4_feature_engineering(train_clean, test_clean, insights)
    best_model, model_results = agent5_modeling(X, y)
    best_model = agent6_validation(best_model, X, y)
    submission = agent7_submission(best_model, X_test, test_clean, sample)