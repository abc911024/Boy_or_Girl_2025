"""
Main script to run all agents in sequence.
"""

from agent.agent1_data_audit import load_data, agent1_data_audit
from agent.agent2_data_cleaning import agent2_data_cleaning
from agent.agent3_eda import agent3_eda
from agent.agent4_feature_engineering import agent4_feature_engineering
from agent.agent5_modeling import agent5_modeling
from agent.agent6_validation import agent6_validation
from agent.agent7_submission import agent7_submission


def main():
    """Run all agents in sequence."""
    print("Starting Multi-Agent Data Science Team...")

    # Load raw data
    train, test, sample = load_data()

    # Agent 1: Data Audit
    train, test = agent1_data_audit(train, test)

    # Agent 2: Data Cleaning
    train_clean, test_clean = agent2_data_cleaning(train, test)

    # Agent 3: EDA
    train_clean, insights = agent3_eda(train_clean)

    # Agent 4: Feature Engineering
    X, y, X_test, feature_info = agent4_feature_engineering(train_clean, test_clean, insights)

    # Agent 5: Modeling
    best_model, model_results, best_model_name = agent5_modeling(X, y)

    # Agent 6: Validation
    validation_results = agent6_validation(best_model, X, y)

    # Agent 7: Submission
    submission_path = agent7_submission(best_model, X_test, test)
    print("\n=== Final Summary ===")
    print(f"A. 專案流程總覽: 資料檢查 → 清洗 → EDA → 特徵工程 → 建模 → 驗證 → 提交")
    print(f"B. 最佳模型: {best_model_name}")
    print(f"C. 模型比較結果: {model_results}")
    print(f"D. 驗證結果: {validation_results}")
    print(f"E. 特徵資訊: {feature_info}")
    print(f"F. 提交檔案位置: {submission_path}")
    print("G. 競賽完成: submission.csv 已準備好，可上傳至 Kaggle。")


if __name__ == "__main__":
    main()