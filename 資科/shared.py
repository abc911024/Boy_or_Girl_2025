"""
Shared functions for all agents.
核心改進：類別補值改為基於模型的補值策略（Model-Based Imputation）
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
    """載入原始資料集"""
    global train, test, sample
    train = pd.read_csv('data/boy or girl 2025 train_missingValue.csv')
    test = pd.read_csv('data/boy or girl 2025 test no ans_missingValue.csv')
    sample = pd.read_csv('data/Boy_or_girl_test_sandbox_sample_submission.csv')

def agent1_data_audit():
    """Agent 1: 資料審計 - 檢查資料結構、缺失值、分佈"""
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

def create_text_features(df):
    """
    為 self_intro 創建文字相關特徵，用於模型補值時的特徵輸入。
    
    特徵說明：
    - self_intro_len: 文字長度
    - self_intro_alpha_ratio: 英文字母佔比
    - self_intro_digit_ratio: 數字佔比
    """
    df['self_intro_len'] = df['self_intro'].fillna('').str.len()
    df['self_intro_alpha_ratio'] = df['self_intro'].fillna('').str.count(r'[a-zA-Z]') / (df['self_intro_len'] + 1)
    df['self_intro_digit_ratio'] = df['self_intro'].fillna('').str.count(r'\d') / (df['self_intro_len'] + 1)
    return df

def impute_categorical_with_model(df_train, df_test, cat_col, num_cols):
    """
    使用 RandomForestClassifier 補值類別欄位缺失值。
    
    核心概念：
    =========
    1. 訓練集中找出該欄位的非缺失列作為訓練資料
    2. 用其他欄位（數值+衍生特徵）作為特徵
    3. 建立分類模型預測該欄位
    4. 對訓練集和測試集中的缺失值進行補值
    
    為什麼比眾數補值更好：
    ====================
    - 眾數補值：無視其他特徵的資訊，補值結果單一
    - 模型補值：考慮其他欄位的相關性，補值結果多樣化
    - 例：高身高男性可能更傾向於某些星座，模型可以捕捉這種關係
    
    Data Leakage 防護：
    ==================
    - 訓練資料（df_train）使用「訓練集本身」建立模型，保證無洩露
    - 測試資料（df_test）使用「訓練集建立的模型」進行預測，無回溯風險
    - 分類模型僅根據訓練集的非缺失列進行 fit，從不看到測試資訊
    
    參數：
    ====
    - df_train: 訓練集 (有 gender 和 cat_col)
    - df_test: 測試集 (無 gender，但有 cat_col)
    - cat_col: 要補值的類別欄位名稱（'star_sign' 或 'phone_os'）
    - num_cols: 用於模型輸入的數值特徵列表
    
    返回：
    ====
    - (df_train, df_test): 補值後的訓練集和測試集
    """
    
    # 檢查該欄位是否有缺失值
    missing_mask_train = df_train[cat_col].isnull()
    missing_mask_test = df_test[cat_col].isnull()
    
    if not (missing_mask_train.any() or missing_mask_test.any()):
        print(f"  [{cat_col}] 無缺失值，跳過")
        return df_train, df_test
    
    # 訓練資料：非缺失列
    train_non_missing = df_train[~missing_mask_train].copy()
    
    # 確保有足夠的訓練樣本（至少 10 個）
    if len(train_non_missing) < 10:
        print(f"  [{cat_col}] 訓練樣本過少 ({len(train_non_missing)} < 10)，使用 'Unknown' 補值")
        df_train.loc[missing_mask_train, cat_col] = 'Unknown'
        df_test.loc[missing_mask_test, cat_col] = 'Unknown'
        return df_train, df_test
    
    # 檢查類別數量是否足夠
    n_classes = train_non_missing[cat_col].nunique()
    if n_classes < 2:
        print(f"  [{cat_col}] 類別數量過少 ({n_classes} < 2)，使用 'Unknown' 補值")
        df_train.loc[missing_mask_train, cat_col] = 'Unknown'
        df_test.loc[missing_mask_test, cat_col] = 'Unknown'
        return df_train, df_test
    
    print(f"  [{cat_col}] 訓練樣本: {len(train_non_missing)}, 缺失值: {missing_mask_train.sum()} (train) + {missing_mask_test.sum()} (test)")
    
    # 準備訓練資料的特徵和標籤
    X_train_impute = train_non_missing[num_cols].copy()
    y_train_impute = train_non_missing[cat_col].copy()
    
    # 建立 RandomForestClassifier 用於補值
    # 參數說明：
    # - n_estimators=50: 50 棵決策樹，足以捕捉複雜關係但不過度複雜
    # - max_depth=10: 深度限制，防止過擬合
    # - random_state=42: 可重複性
    # - n_jobs=-1: 平行運算
    imputer_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    imputer_model.fit(X_train_impute, y_train_impute)
    
    # 補值訓練集中的缺失值
    if missing_mask_train.any():
        X_missing_train = df_train.loc[missing_mask_train, num_cols].copy()
        predictions_train = imputer_model.predict(X_missing_train)
        df_train.loc[missing_mask_train, cat_col] = predictions_train
    
    # 補值測試集中的缺失值
    if missing_mask_test.any():
        X_missing_test = df_test.loc[missing_mask_test, num_cols].copy()
        predictions_test = imputer_model.predict(X_missing_test)
        df_test.loc[missing_mask_test, cat_col] = predictions_test
    
    return df_train, df_test

def clean_df(df, is_train=True):
    """
    數據清理的主函數（不進行類別補值，由 clean_df_with_model_imputation 統一管理）。
    
    步驟說明：
    ========
    1. 將 'yt' 轉換為數值
    2. 檢測並移除數值欄位中的異常值（使用 IQR 方法）
    3. 用中位數補值數值欄位
    4. 創建文字特徵
    5. 處理特殊值和數據型別
    """
    
    # 步驟 1: 轉換 'yt' 為數值型
    df['yt'] = pd.to_numeric(df['yt'], errors='coerce')
    
    # 步驟 2: 異常值檢測與處理（使用 IQR 方法）
    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']
    for col in num_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            # 將異常值設為 NaN，後續用中位數補值
            df[col] = np.where((df[col] < lower) | (df[col] > upper), np.nan, df[col])
    
    # 步驟 3: 數值欄位中位數補值
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # 步驟 4: 創建文字特徵（用於類別補值模型）
    df = create_text_features(df)
    
    # 步驟 5: 處理特殊值
    df['self_intro'] = df['self_intro'].fillna('')
    df['phone_os'] = df['phone_os'].replace('JohnCena', 'Android')
    
    # 步驟 6: 設置數據型別
    df['id'] = df['id'].astype(int)
    if is_train:
        df['gender'] = df['gender'].astype(int)
    df[['star_sign', 'phone_os', 'self_intro']] = df[['star_sign', 'phone_os', 'self_intro']].astype(str)
    df[num_cols] = df[num_cols].astype(float)
    
    return df

def clean_df_with_model_imputation(df_train, df_test):
    """
    整合數據清理和模型基礎補值的主函數。
    
    工作流程：
    ========
    1. 分別清理訓練集和測試集（數值補值、文字特徵）
    2. 對類別欄位使用模型補值
    3. 返回補值完成的訓練集和測試集
    
    Data Leakage 防護：
    ==================
    - 訓練集和測試集分開處理，模型只在訓練集上 fit
    - 類別補值模型使用訓練集的資訊，避免測試集訊息洩露
    - 縮放器、編碼器等只在訓練集上 fit，應用於測試集
    """
    
    # 分別清理兩個資料集
    df_train = clean_df(df_train, is_train=True)
    df_test = clean_df(df_test, is_train=False)
    
    # 用於補值模型的特徵列表
    feature_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt',
                    'self_intro_len', 'self_intro_alpha_ratio', 'self_intro_digit_ratio']
    
    # 補值 'star_sign'
    print("補值 star_sign...")
    df_train, df_test = impute_categorical_with_model(df_train, df_test, 'star_sign', feature_cols)
    
    # 補值 'phone_os'
    print("補值 phone_os...")
    df_train, df_test = impute_categorical_with_model(df_train, df_test, 'phone_os', feature_cols)
    
    return df_train, df_test

def agent2_data_cleaning():
    """
    Agent 2: 數據清理與模型基礎缺失值補值
    
    核心改進：
    ========
    - 從「眾數補值」改為「模型補值」
    - 類別欄位 (star_sign, phone_os) 使用 RandomForestClassifier 預測
    - 數值欄位保留中位數補值
    - 完整避免 data leakage
    
    補值策略比較：
    ============
    
    眾數補值（舊方法）：
    - 優點：簡單快速
    - 缺點：無視其他特徵關係，可能降低模型表現
    
    模型補值（新方法）：
    - 優點：考慮特徵間相關性，補值更準確，提升下游模型表現
    - 缺點：計算量稍大，但完全值得
    """
    global train, test, train_clean, test_clean
    print("=== Agent 2: Data Cleaning (Model-Based Imputation) ===")
    
    # 使用新的整合清理與補值函數
    train_clean, test_clean = clean_df_with_model_imputation(train.copy(), test.copy())
    
    # 驗證缺失值已完全補值
    train_missing = train_clean.isnull().sum().sum()
    test_missing = test_clean.isnull().sum().sum()
    print(f"Train clean missing: {train_missing}")
    print(f"Test clean missing: {test_missing}")
    
    # 保存清理後的資料集（不修改原始文件）
    train_clean.to_csv('data/train_cleaned.csv', index=False)
    test_clean.to_csv('data/test_cleaned.csv', index=False)
    print("Cleaned datasets saved to data/train_cleaned.csv and data/test_cleaned.csv")

def agent3_eda():
    """
    Agent 3: EDA (探索性資料分析)
    
    分析對象：
    - 數值欄位與目標變數的關係 (t-test)
    - 類別欄位與目標變數的關係 (chi-square test)
    - 衍生特徵的統計意義
    """
    global train_clean, insights
    print("=== Agent 3: EDA ===")
    target = 'gender'
    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']
    cat_cols = ['star_sign', 'phone_os']

    # 數值欄位 t-test
    print("\n數值欄位 t-test (性別差異檢驗):")
    for col in num_cols:
        group1 = train_clean[train_clean[target] == 1][col]
        group2 = train_clean[train_clean[target] == 2][col]
        t_stat, p = ttest_ind(group1, group2, nan_policy='omit')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {col}: p={p:.4f} {sig}")

    # 類別欄位 chi-square test
    print("\n類別欄位 chi-square test (性別獨立性檢驗):")
    for col in cat_cols:
        contingency = pd.crosstab(train_clean[col], train_clean[target])
        chi2, p, _, _ = chi2_contingency(contingency)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {col}: chi2={chi2:.2f}, p={p:.4f} {sig}")

    # 文字特徵分析
    print("\n文字特徵 t-test:")
    if 'self_intro_len' not in train_clean.columns:
        train_clean['self_intro_len'] = train_clean['self_intro'].str.len()
    len1 = train_clean[train_clean[target] == 1]['self_intro_len']
    len2 = train_clean[train_clean[target] == 2]['self_intro_len']
    t_len, p_len = ttest_ind(len1, len2, nan_policy='omit')
    sig = "***" if p_len < 0.001 else "**" if p_len < 0.01 else "*" if p_len < 0.05 else "ns"
    print(f"  self_intro_len: p={p_len:.4f} {sig}")

    # BMI 衍生特徵
    if 'bmi' not in train_clean.columns:
        train_clean['bmi'] = train_clean['weight'] / (train_clean['height'] / 100) ** 2
    bmi1 = train_clean[train_clean[target] == 1]['bmi']
    bmi2 = train_clean[train_clean[target] == 2]['bmi']
    t_bmi, p_bmi = ttest_ind(bmi1, bmi2, nan_policy='omit')
    sig = "***" if p_bmi < 0.001 else "**" if p_bmi < 0.01 else "*" if p_bmi < 0.05 else "ns"
    print(f"  bmi: p={p_bmi:.4f} {sig}")

    # 保存分析結果
    insights = {
        'significant_num': [col for col in num_cols if ttest_ind(train_clean[train_clean[target] == 1][col], train_clean[train_clean[target] == 2][col], nan_policy='omit')[1] < 0.05],
        'significant_cat': [col for col in cat_cols if chi2_contingency(pd.crosstab(train_clean[col], train_clean[target]))[1] < 0.05],
        'text_features': ['self_intro_len', 'bmi']
    }
    print(f"\n統計顯著特徵 (p < 0.05):\n  數值: {insights['significant_num']}\n  類別: {insights['significant_cat']}")

def agent4_feature_engineering():
    """
    Agent 4: 特徵工程
    
    流程：
    1. OneHotEncoding 類別欄位 (star_sign, phone_os)
    2. 衍生特徵 (self_intro_len, self_intro_alpha_ratio, bmi 等)
    3. StandardScaler 正規化數值特徵
    4. 準備 X, y, X_test 用於模型訓練
    """
    global train_clean, test_clean, X, y, X_test
    print("=== Agent 4: Feature Engineering ===")

    cat_cols = ['star_sign', 'phone_os']
    
    # OneHotEncoding 類別欄位
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_train = encoder.fit_transform(train_clean[cat_cols])
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols))
    train_clean = pd.concat([train_clean.reset_index(drop=True), encoded_train_df], axis=1)
    train_clean.drop(cat_cols, axis=1, inplace=True)

    encoded_test = encoder.transform(test_clean[cat_cols])
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols))
    test_clean = pd.concat([test_clean.reset_index(drop=True), encoded_test_df], axis=1)
    test_clean.drop(cat_cols, axis=1, inplace=True)

    # 補充衍生特徵
    for df in [train_clean, test_clean]:
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

    # 數值特徵列表（包含新補充的特徵）
    num_cols = ['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 
                'self_intro_len', 'self_intro_alpha_ratio', 'self_intro_digit_ratio', 'bmi']
    
    # 特徵正規化（StandardScaler）
    scaler = StandardScaler()
    train_clean[num_cols] = scaler.fit_transform(train_clean[num_cols])
    test_clean[num_cols] = scaler.transform(test_clean[num_cols])

    # 準備訓練和預測集
    X = train_clean[num_cols]
    y = train_clean['gender']
    X_test = test_clean[num_cols]

    print(f"Final feature count: {X.shape[1]}")
    print(f"Training set shape: {X.shape}")
    print(f"Test set shape: {X_test.shape}")

def agent5_modeling():
    """
    Agent 5: 模型訓練
    
    比較多個基線模型，選擇驗證準確率最高的模型進行最終訓練。
    """
    global X, y, best_model, model_results
    print("=== Agent 5: Modeling ===")

    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
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
    print(f"\n選擇最佳模型: {best_model_name}")

def agent6_validation():
    """
    Agent 6: 驗證
    
    使用 Stratified 5-Fold Cross-Validation 評估模型性能
    """
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
    """
    Agent 7: 提交檔案生成
    
    使用最佳模型對測試集進行預測，生成提交檔案
    """
    global best_model, X_test, test_clean, sample, submission
    print("=== Agent 7: Submission ===")

    y_pred = best_model.predict(X_test)
    submission = pd.DataFrame({'id': test_clean['id'], 'gender': y_pred})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
    print(f"Sample submission shape: {sample.shape}")
    print(f"Our submission shape: {submission.shape}")
    print(f"Gender distribution:\n{submission['gender'].value_counts()}")
