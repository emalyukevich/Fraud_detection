import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

def get_important_features_by_corr(df, target_col='Class', threshold=0.1, exclude_target=True, verbose=False):
    """
    Возвращает список признаков с модулем корреляции ≥ threshold по отношению к целевой переменной.
    
    Parameters:
    - df (pd.DataFrame): датафрейм с данными
    - target_col (str): название целевой переменной
    - threshold (float): порог по модулю корреляции
    - exclude_target (bool): исключать ли целевую переменную из результата
    - verbose (bool): печатать ли отсортированные корреляции

    Returns:
    - List[str]: список названий признаков
    """
    correlations = df.corr()[target_col].drop(target_col)
    important_features = correlations[correlations.abs() >= threshold].sort_values(key=abs, ascending=False)

    if verbose:
        display(important_features)

    return important_features.index.tolist() if exclude_target else important_features

def find_best_threshold(y_true, scores, target_recall, direction='lower', verbose=False):
    """
    Находит оптимальный порог для бинарной классификации на основе метрик precision, recall и F1.
    Минимизирует false positives при достижении целевого recall.
    
    Parameters:
    - y_true (array-like): истинные метки классов (1 - положительный класс)
    - scores (array-like): предсказанные scores/аномалии (чем меньше, тем более аномально)
    - target_recall (float): целевой уровень recall, который необходимо достичь
    - verbose (bool): печатать ли информацию о процессе поиска
    
    Returns:
    - dict: словарь с лучшим порогом и метриками:
        {
            'threshold': float, 
            'precision': float,
            'recall': float,
            'f1': float
        }
    """
    thresholds = np.linspace(np.min(scores), np.max(scores), 1000)
    best = {'threshold': None, 'precision': 0, 'recall': 0, 'f1': 0}
    
    for t in thresholds:
        if direction == 'lower':
            preds = (scores < t).astype(int)
        elif direction == 'higher':
            preds = (scores >= t).astype(int)
        else:
            raise ValueError("Direction must be 'lower' or 'higher'")
        
        if np.sum(preds) == 0:
            continue
            
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)

        if r >= target_recall and f1 > best['f1']:
            best.update({'threshold': t, 'precision': p, 'recall': r, 'f1': f1})
            
            if verbose:
                print(f"New best: Thresh={t:.3f} | Prec={p:.3f} | Rec={r:.3f} | F1={f1:.3f}")

    return best

def add_anomaly_score(X, ocsvm_model):
    """
    Добавляет оценку аномальности (anomaly_score) к данным с помощью обученной OneClassSVM модели.
    Возвращает копию DataFrame с новым признаком 'anomaly_score'.

    Parameters:
    - X (pd.DataFrame)
    - ocsvm_model (sklearn.OneClassSVM): Обученная модель OneClassSVM.

    Returns:
    - pd.DataFrame: Копия входных данных с добавленным столбцом 'anomaly_score'.
    """
    X = X.copy()
    X['anomaly_score'] = ocsvm_model.decision_function(X)
    return X
