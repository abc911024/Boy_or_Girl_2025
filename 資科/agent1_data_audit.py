"""
Agent 1: Data Audit - Quality check for train and test data.
"""

from shared import load_data, agent1_data_audit

if __name__ == "__main__":
    load_data()
    agent1_data_audit()