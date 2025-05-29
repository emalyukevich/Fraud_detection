# 💳 Credit Card Fraud Detection — Advanced ML Project

Проект по выявлению мошеннических транзакций с использованием **unsupervised learning**, **ансамблей моделей** и **оптимизации порога классификации**. Структура и подходы ориентированы на реальный банковский use-case.

---

## 📌 Цели проекта

- 📉 Выявить мошеннические транзакции в условиях сильного дисбаланса классов  
- 🔍 Использовать **One-Class SVM** как основу для поиска аномалий  
- 🧠 Собрать **ансамбль моделей** для усиления предсказаний  
- 🛡 Минимизировать **ложноотрицательные (False Negatives)**  
- 🧱 Реализовать **модульную архитектуру**, пригодную для деплоя и масштабирования

---

## 🗂️ Структура проекта

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

## 🧠 Методы и инструменты

| Категория           | Используемые методы                                             |
|---------------------|----------------------------------------------------------------|
| 📊 Exploratory Data Analysis | `Boxplot`, Стратифицированный sampling, `UMAP`, Корреляционный анализ |
| 🏗 Feature Engineering | `StandardScaler`, Отбор признаков, Анализ важности            |
| 🕵 Anomaly Detection | `One-Class SVM`                                                |
| 🤖 Модели классификации | `KNN`, `RandomForestClassifier`, `Logistic Regression`      |
| 🔁 Ансамблирование     | Усреднение вероятностей, Оптимизация порога                  |
| 📏 Метрики            | `Recall`, `Precision`, `F1`, `ROC-AUC`, `Confusion Matrix`   |

---

## 📊 Результаты

| Метрика                | Значение |
|------------------------|----------|
| **Recall (Fraud)**     | 0.82     |
| **Precision (Fraud)**  | 0.86     |
| **F1-score (Fraud)**   | 0.84     |
| **ROC AUC (общий)**    | 0.96     |
| **False Negatives**    | 13       |
| **False Positives**    | 10       |

> 📌 Модель показывает высокую отзывчивость (recall) и отличную AUC даже при сильном дисбалансе (fraud < 0.2%).

---

## 🚀 Быстрый старт

1. Установи зависимости:
```bash
git clone https://github.com/your-username/creditcard-fraud-detection.git
cd creditcard-fraud-detection
pip install -r requirements.txt

2. Обработай данные:
jupyter notebook notebooks/02_Feature_Engineering.ipynb

3. Обучи ансамбль:
jupyter notebook notebooks/03_Baseline_Classification_Model.ipynb

4. Проверь финальные метрики:
jupyter notebook notebooks/04_Final_Model_Evaluation.ipynb
```
---

## 🔭 Возможности для расширения

### 🧠 Модели и гиперпараметры
- 🎛 Реализовать **тонкую настройку гиперпараметров** моделей:
  - `LogisticRegression` — `penalty`, `C`, `solver`
  - `KNeighborsClassifier` — `n_neighbors`, `weights`, `metric`
  - `RandomForestClassifier` — `n_estimators`, `max_depth`, `class_weight`
- 🧪 Подключить **GridSearchCV** или `Optuna` для автоматического подбора параметров

### ⏱ Моделирование временных паттернов
- 🧩 Построить **Time-aware модель**, учитывающую:
  - Временные интервалы между транзакциями (Time-delta features)
  - Сезонные/суточные паттерны
  - Скользящие окна (`rolling mean`, `lag features`)
- 📈 Построить **временные профили клиентов**

### 📊 Интерпретируемость
- 🧠 Подключить **SHAP** для глобального и локального объяснения решений модели
- 🧪 Добавить **LIME** для точечных интерпретаций на тестовых примерах
- 💡 Генерировать отчёты "почему транзакция считается мошеннической?"

### 🛰 Деплой и API
- 🚀 Реализовать **FastAPI-сервер**:
  - Загрузка обученной модели
  - Предсказания по REST API
  - Визуализация метрик и логов в Swagger UI

### 🧮 Метрики и оценка
- 📉 Обновить фокус с ROC AUC на **PR AUC** (Precision-Recall Area Under Curve)
- ⚖ Особенно актуально при **экстремальном дисбалансе классов**
- 📌 Добавить cost-based метрики: например, оценка стоимости FN vs FP

### 🏷 Feature Engineering
- 🧬 Использовать больше признаков на базе `OneClassSVM`:
  - `is_anomaly` как бинарный флаг
  - `decision_function` как `anomaly_score`
  - `rank` по степени аномальности
- 🔬 Более тщательно проанализировать переменные:
  - `Time` — ввести признаки `hour`, `delta_time`, `time_since_last_tx`
  - `Amount` — логарифмирование, масштабирование, binning

---
