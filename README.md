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

---

## 🧠 Методы и инструменты

| Категория           | Используемые методы                                             |
|---------------------|----------------------------------------------------------------|
| 📊 Exploratory Data Analysis | `Boxplot`, `t-SNE`, `PCA`, корреляция признаков |
| 🏗 Feature Engineering | `RobustScaler`, отбор признаков, анализ важности            |
| 🕵 Anomaly Detection | `One-Class SVM`                                                |
| 🤖 Модели классификации | `MLPClassifier`, `CatBoostClassifier`                      |
| 🔁 Ансамблирование     | Усреднение вероятностей, оптимизация порога                  |
| 📏 Метрики            | `Recall`, `Precision`, `F1`, `ROC AUC`, `Confusion Matrix`   |

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