# 💳 Credit Card Fraud Detection — Advanced ML Project

A project aimed at detecting fraudulent transactions using **unsupervised learning**, **model ensembles**, and **threshold optimization**. The structure and methods are tailored to real-world banking use cases.

---

---

## 📌 Project Goals

- 📉 Detect fraudulent transactions under extreme class imbalance  
- 🔍 Leverage **One-Class SVM** for anomaly detection  
- 🧠 Build a **model ensemble** to enhance predictions  
- 🛡 Minimize **False Negatives (FN)** to prevent fraud losses  
- 🧱 Implement a **modular architecture** suitable for deployment and scaling

---

## 🗂️ Project Structure

```bash
project-root/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_Feature_Engineering.ipynb
│ ├── 03_Baseline_Classification_Model.ipynb
│ └── 04_Final_Model_Evaluation.ipynb
├── src/
│ ├── init.py
│ ├── utils.py
│ └── models/
│ ├── anomaly_detection.ipynb
│ ├── ensemble.py
│ ├── ensemble_model.joblib
│ └── oneclasssvm_anomaly_detector.joblib
├── README.md
└── .gitignore

```

---

## 🧠 Methods & Tools

| Category                | Techniques and Tools Used                                      |
|------------------------|---------------------------------------------------------------|
| 📊 Exploratory Analysis | `Boxplot`, Stratified sampling, `UMAP`, Correlation analysis  |
| 🏗 Feature Engineering  | `StandardScaler`, Feature selection, Feature importance        |
| 🕵 Anomaly Detection    | `One-Class SVM`                                                |
| 🤖 Classification       | `KNN`, `RandomForestClassifier`, `LogisticRegression`          |
| 🔁 Ensembling           | Averaging probabilities, Threshold optimization               |
| 📏 Evaluation Metrics   | `Recall`, `Precision`, `F1`, `ROC-AUC`, `Confusion Matrix`     |

---

## 📊 Results

| Metric                   | Value  |
|--------------------------|--------|
| **Recall (Fraud)**       | 0.82   |
| **Precision (Fraud)**    | 0.86   |
| **F1-score (Fraud)**     | 0.84   |
| **ROC AUC (Overall)**    | 0.96   |
| **False Negatives (FN)** | 13     |
| **False Positives (FP)** | 10     |

> 📌 The model demonstrates high recall and AUC despite the class imbalance (fraud < 0.2%).

---

## 🚀 Quick Start

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/creditcard-fraud-detection.git
cd creditcard-fraud-detection
pip install -r requirements.txt

2. Preprocess the data:
jupyter notebook notebooks/02_Feature_Engineering.ipynb

3. Train the ensemble:
jupyter notebook notebooks/03_Baseline_Classification_Model.ipynb

4. Evaluate final metrics:
jupyter notebook notebooks/04_Final_Model_Evaluation.ipynb
```
---

## 🔭 Future Enhancements

### 🧠 Model and Hyperparameter Tuning

- 🎛 Implement **fine-tuning of model hyperparameters**:
  - `LogisticRegression`: `penalty`, `C`, `solver`
  - `KNeighborsClassifier`: `n_neighbors`, `weights`, `metric`
  - `RandomForestClassifier`: `n_estimators`, `max_depth`, `class_weight`
- 🧪 Use **GridSearchCV** or `Optuna` for automated parameter optimization

### ⏱ Temporal Pattern Modeling

- 🧩 Develop a **time-aware model** that accounts for:
  - Time intervals between transactions (`time_delta` features)
  - Seasonal and daily patterns
  - Rolling statistics (`rolling mean`, `lag features`)
- 📈 Build **customer behavior profiles** over time

### 📊 Interpretability

- 🧠 Integrate **SHAP** for both global and local interpretability
- 🧪 Add **LIME** for instance-specific explanations
- 💡 Generate clear reports explaining why a transaction was flagged as fraudulent

### 🛰 Deployment and API

- 🚀 Build a **FastAPI server** to:
  - Load the trained model
  - Serve predictions via REST API
  - Display metrics and logs via Swagger UI

### 🧮 Metrics and Evaluation

- 📉 Shift the focus from ROC AUC to **PR AUC (Precision-Recall AUC)**, more suitable for imbalanced data
- ⚖ Incorporate **cost-sensitive metrics**, such as evaluating the cost of False Negatives (FN) vs False Positives (FP)

### 🏷 Feature Engineering

- 🧬 Derive additional features from `One-Class SVM`:
  - `is_anomaly`: binary anomaly flag
  - `decision_function`: continuous anomaly score
  - `rank`: anomaly ranking based on score
- 🔬 Further enhance features like:
  - `Time`: extract `hour`, `delta_time`, `time_since_last_tx`
  - `Amount`: apply log transformation, scaling, and binning

---