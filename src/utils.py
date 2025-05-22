import pandas as pd
import numpy as np

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
