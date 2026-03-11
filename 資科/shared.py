"""
Shared functions for all agents.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import ttest_ind, chi2_contingency

# Global variables to store data
train = None
test = None
sample = None
train_clean = None
test_clean = None
insights = None
X = None
y = None
X_test = None
best_model = None
model_results = None
submission = None

def load_data():
    global train, test, sample
    train = pd.read_csv('data/boy or girl 2025 train_missingValue.csv')
    test = pd.read_csv('data/boy or girl 2025 test no ans_missingValue.csv')
    sample = pd.read_csv('data/Boy_or_girl_test_sandbox_sample_submission.csv')

def agent1_data_audit():
    global train, test
    print("=== Agent 1: Data Audit ===")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Train dtypes:\n{train.dtypes}")
    print(f"Test dtypes:\n{test.dtypes}")
    print(f"Train missing:\n{train.isnull().sum()}")
    print(f"Test missing:\n{test.isnull().sum()}")
    print(f"Gender distribution:\n{train['gender'].value_counts()}")
    print(f"Star_sign unique: {sorted(train['star_sign'].dropna().unique())}")
    print(f"Phone_os unique: {sorted(train['phone_os'].dropna().unique())}")
    print(f"YT sample: {train['yt'].dropna().unique()[:10]}")
    print(f"Self_intro sample: {train['self_intro'].dropna().head(5).tolist()}")
    print(f"Numerical describe:\n{train[['height', 'weight', 'sleepiness', 'iq', 'fb_friends']].describe()}")
    print(f"Duplicates: {train.duplicated().sum()}, ID duplicates: {train['id'].duplicated().sum()}")

def clean_df(df, is_train=True):
    df['yt'] = pd.to_numeric(df['yt'], errors='coerce')
    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']
    for col in num_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.where((df[col] < lower) | (df[col] > upper), np.nan, df[col])
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[['star_sign', 'phone_os']] = cat_imputer.fit_transform(df[['star_sign', 'phone_os']])
    df['self_intro'] = df['self_intro'].fillna('')
    df['phone_os'] = df['phone_os'].replace('JohnCena', 'Android')
    df['id'] = df['id'].astype(int)
    if is_train:
        df['gender'] = df['gender'].astype(int)
    df[['star_sign', 'phone_os', 'self_intro']] = df[['star_sign', 'phone_os', 'self_intro']].astype(str)
    df[num_cols] = df[num_cols].astype(float)
    return df

def agent2_data_cleaning():
    global train, test, train_clean, test_clean
    print("=== Agent 2: Data Cleaning ===")
    train_clean = clean_df(train, True)
    test_clean = clean_df(test, False)
    print(f"Train clean missing: {train_clean.isnull().sum().sum()}")
    print(f"Test clean missing: {test_clean.isnull().sum().sum()}")

def agent3_eda():
    global train_clean, insights
    print("=== Agent 3: EDA ===")
    target = 'gender'
    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']
    cat_cols = ['star_sign', 'phone_os']

    for col in num_cols:
        group1 = train_clean[train_clean[target] == 1][col]
        group2 = train_clean[train_clean[target] == 2][col]
        t_stat, p = ttest_ind(group1, group2, nan_policy='omit')
        print(f"{col}: p={p:.4f}")

    for col in cat_cols:
        contingency = pd.crosstab(train_clean[col], train_clean[target])
        chi2, p, _, _ = chi2_contingency(contingency)
        print(f"{col}: chi2={chi2:.2f}, p={p:.4f}")

    train_clean['self_intro_len'] = train_clean['self_intro'].str.len()
    len1 = train_clean[train_clean[target] == 1]['self_intro_len']
    len2 = train_clean[train_clean[target] == 2]['self_intro_len']
    t_len, p_len = ttest_ind(len1, len2, nan_policy='omit')
    print(f"Self_intro_len: p={p_len:.4f}")

    train_clean['bmi'] = train_clean['weight'] / (train_clean['height'] / 100) ** 2
    bmi1 = train_clean[train_clean[target] == 1]['bmi']
    bmi2 = train_clean[train_clean[target] == 2]['bmi']
    t_bmi, p_bmi = ttest_ind(bmi1, bmi2, nan_policy='omit')
    print(f"BMI: p={p_bmi:.4f}")

    insights = {
        'significant_num': [col for col in num_cols if ttest_ind(train_clean[train_clean[target] == 1][col], train_clean[train_clean[target] == 2][col], nan_policy='omit')[1] < 0.05],
        'significant_cat': [col for col in cat_cols if chi2_contingency(pd.crosstab(train_clean[col], train_clean[target]))[1] < 0.05],
        'text_features': ['self_intro_len', 'bmi']
    }

def agent4_feature_engineering():
    global train_clean, test_clean, X, y, X_test
    print("=== Agent 4: Feature Engineering ===")

    cat_cols = ['star_sign', 'phone_os']
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_train = encoder.fit_transform(train_clean[cat_cols])
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols))
    train_clean = pd.concat([train_clean, encoded_train_df], axis=1)
    train_clean.drop(cat_cols, axis=1, inplace=True)

    encoded_test = encoder.transform(test_clean[cat_cols])
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols))
    test_clean = pd.concat([test_clean, encoded_test_df], axis=1)
    test_clean.drop(cat_cols, axis=1, inplace=True)

    for df in [train_clean, test_clean]:
        df['self_intro_len'] = df['self_intro'].str.len()
        df['self_intro_alpha_ratio'] = df['self_intro'].str.count(r'[a-zA-Z]') / (df['self_intro_len'] + 1)
        df['self_intro_digit_ratio'] = df['self_intro'].str.count(r'\d') / (df['self_intro_len'] + 1)
        df.drop('self_intro', axis=1, inplace=True)
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'self_intro_len', 'self_intro_alpha_ratio', 'self_intro_digit_ratio', 'bmi']
    scaler = StandardScaler()
    train_clean[num_cols] = scaler.fit_transform(train_clean[num_cols])
    test_clean[num_cols] = scaler.transform(test_clean[num_cols])

    X = train_clean[num_cols]
    y = train_clean['gender']
    X_test = test_clean[num_cols]

    print(f"Features: {X.shape[1]}")

def agent5_modeling():
    global X, y, best_model, model_results
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
    model_results = results

def agent6_validation():
    global best_model, X, y
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

def agent7_submission():
    global best_model, X_test, test_clean, sample, submission
    print("=== Agent 7: Submission ===")

    y_pred = best_model.predict(X_test)
    submission = pd.DataFrame({'id': test_clean['id'], 'gender': y_pred})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
    print(f"Sample submission shape: {sample.shape}")
    print(f"Our submission shape: {submission.shape}")
    print(f"Gender distribution: {submission['gender'].value_counts()}")
