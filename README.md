# Multi-Agent Data Science Pipeline

> 透過模組化代理（Agent）分工，實現從資料清理、特徵工程、模型訓練到預測提交的端到端 (End-to-End) 自動化機器學習流程。

## 📖 專案簡介 (Project Overview)

本專案建構了一個**多代理資料科學流程 (Multi-Agent Data Science Pipeline)**。有別於傳統將所有程式碼集中於單一腳本的作法，本系統將完整的資料科學生命週期（Data Science Lifecycle）拆解為 7 個獨立的 Agent 節點。

每個 Agent 專責處理特定的分析或工程任務，不僅大幅提升了程式碼的可讀性與可維護性，更完美契合業界標準的 MLOps 流程。本架構具備高度泛用性，可快速抽換資料集並自動執行完整的模型訓練與驗證管線。

## ✨ 核心技術亮點 (Key Highlights)

* **進階資料修復 (Advanced Imputation)**：實作 Model-based Imputation，利用隨機森林等機器學習演算法，根據其他共變數特徵來預測並填補缺失的類別變數，避免單純依賴眾數或常數填補所造成的偏差。
* **嚴謹的探索性分析 (Statistical EDA)**：運用 T-test 與 Chi-square test 等統計手法進行科學化檢定，並計算 Effect Size (Cohen's d) 量化特徵影響力，以數據驅動取代主觀特徵篩選。
* **多維度特徵工程 (Feature Engineering)**：無縫整合數值衍生變數建構、類別特徵編碼（One-Hot Encoding），以及針對非結構化自我介紹文字的自然語言處理（基於 TF-IDF 演算法的向量化特徵擷取）。
* **穩健的模型驗證 (Robust Validation)**：內建多模型交叉比較機制，並採用 Stratified K-Fold 交叉驗證確保類別分佈均勻，全面評估 Accuracy, Precision, Recall 與 F1-Score 等關鍵指標。

## 🏗️ 系統架構與代理工作流 (Pipeline Architecture)

本系統由以下 7 個專責 Agent 依序執行，形成完整的自動化流水線：

### 1. 🕵️ Agent 1: Data Audit (資料盤點)
* **任務**：載入訓練集 (Train) 與測試集 (Test) 資料。
* **實作**：自動掃描並量化各欄位的缺失值分佈與資料型態，產出初始的資料品質報告，以決定後續清理策略。

### 2. 🧹 Agent 2: Data Cleaning (資料清理)
* **任務**：處理髒數據，確保資料結構健全。
* **實作**：
  * **異常值處理**：運用 IQR (Interquartile Range) 演算法精準偵測並剔除數值型離群值 (Robust Outlier Removal)。
  * **資料校正**：修正不合理的類別標籤與輸入錯誤 (Data Anomaly Correction)。
  * **缺失值填補**：數值型特徵採用中位數 (Median) 補值以抵抗極端值影響；文字型特徵標準化處理；關鍵類別型特徵則採用 Model-based 預測填補技術。

### 3. 📊 Agent 3: Exploratory Data Analysis (EDA)
* **任務**：挖掘特徵與目標變數 (Target Variable) 之間的潛在關聯。
* **實作**：針對數值變數執行 T-test 檢定，針對類別變數執行 Chi-square test。自動標註具備統計顯著性 (p-value < 0.05) 的關鍵特徵，並產出效應值評估。

### 4. ⚙️ Agent 4: Feature Engineering (特徵工程)
* **任務**：將原始資料轉換為機器學習模型可有效利用的高維度特徵空間。
* **實作**：
  * 結合跨欄位資訊，建構具備業務邏輯的複合數學指標 (如比例、加總等衍生變數)。
  * 將類別變數轉換為機器學習友善的 One-Hot Encoding 格式。
  * 針對文字欄位提取長度、字元組成比例等特徵，並運用 **TF-IDF Vectorizer** 將文本轉換為稀疏向量特徵。

### 5. 🧠 Agent 5: Model Training (模型訓練)
* **任務**：訓練並篩選出表現最佳的預測演算法。
* **實作**：導入 K-Fold Cross Validation 橫向評估多種機器學習演算法（預設包含 Logistic Regression, Random Forest, Extra Trees 等），並依據驗證集表現自動選出最佳模型 (Best Estimator)。

### 6. ✅ Agent 6: Model Validation (模型驗證)
* **任務**：確保最終挑選之模型具備良好的泛化能力，避免過度擬合 (Overfitting)。
* **實作**：對最佳模型執行深度 Stratified K-Fold 交叉驗證，綜合評估 Accuracy, F1 Score, Precision 與 Recall，產出整體預測成效指標。

### 7. 🚀 Agent 7: Submission (預測與提交)
* **任務**：產出最終預測結果。
* **實作**：載入訓練完成且經驗證的最佳模型，對未知的 Test Dataset 進行最終推論，並依據標準格式匯出預測結果檔案 (`submission.csv`) 供後續評估或系統介接使用。

## 🛠️ 技術堆疊 (Tech Stack)

* **語言**: Python 3.x
* **資料處理與運算**: Pandas, NumPy
* **機器學習模型與管道**: Scikit-Learn
* **統計分析**: SciPy, Statsmodels

## 📁 專案結構 (Project Structure)

```text
├── data/                   # 存放原始 Train, Test 與 Submission 樣本檔案
├── agents/                 # 放置各階段獨立 Agent 的模組化程式碼
│   ├── agent1_audit.py
│   ├── agent2_cleaning.py
│   ├── agent3_eda.py
│   └── ...
├── main.py                 # 驅動所有 Agent 依序執行之主程式
├── requirements.txt        # 專案相依 Python 套件清單
├── README.md               # 專案說明文件
└── output/                 # 存放最終預測結果 (submission.csv) 與產出模型
