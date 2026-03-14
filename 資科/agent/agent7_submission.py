"""
Agent 7: Submission - Generate final prediction file.
"""

import pandas as pd


def agent7_submission(best_model, X_test, test_df, output_path="submission.csv"):
    """Generate submission file with same number of rows as test."""
    print("=== Agent 7: Submission ===")

    preds = best_model.predict(X_test)

    if len(preds) != len(test_df):
        raise ValueError(
            f"預測筆數 {len(preds)} 和 test 筆數 {len(test_df)} 不一致"
        )

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "gender": preds
    })

    submission.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Submission file saved to: {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())

    return output_path