"""
Agent 1: Data Audit
"""

from pathlib import Path
import pandas as pd

from pathlib import Path
import pandas as pd

from pathlib import Path
import pandas as pd


def load_data():
    """Load train, test, and sample submission data."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    train_path = data_dir / "boy or girl 2025 train_missingValue.csv"
    test_path = data_dir / "boy or girl 2025 test no ans_missingValue.csv"
    sample_path = data_dir / "Boy_or_girl_test_sandbox_sample_submission.csv"

    for path in [train_path, test_path, sample_path]:
        if not path.exists():
            raise FileNotFoundError(f"找不到檔案: {path}")

    print("Loading files from:")
    print(train_path)
    print(test_path)
    print(sample_path)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    return train, test, sample

def agent1_data_audit(train, test):
    """Basic data audit."""
    print("=== Agent 1: Data Audit ===")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nTrain missing values:")
    print(train.isnull().sum())
    print("\nTest missing values:")
    print(test.isnull().sum())
    return train, test