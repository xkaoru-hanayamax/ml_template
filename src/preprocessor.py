import pandas as pd
from typing import Tuple

from src.config import DatasetConfig


def preprocess_train(df: pd.DataFrame, config: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    訓練データの前処理: 不要列削除 + ターゲット分離

    Args:
        df: 訓練データのDataFrame
        config: データセット設定

    Returns:
        X: 特徴量DataFrame
        y: ターゲットSeries
    """
    df = df.copy()

    # ターゲット列を分離
    y = df[config.target_col]

    # 削除列リストを構築（ID列 + ターゲット列 + 設定で指定された削除列）
    drop_cols = [config.id_col, config.target_col] + config.drop_cols
    X = df.drop(columns=drop_cols)

    # カテゴリ変数をcategory型に変換（LightGBM用）
    for col in config.categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
        else:
            print(f"Warning: Categorical column '{col}' not found in features")

    return X, y


def preprocess_test(df: pd.DataFrame, config: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    テストデータの前処理: ID保持 + 不要列削除

    Args:
        df: テストデータのDataFrame
        config: データセット設定

    Returns:
        X: 特徴量DataFrame
        ids: ID列のSeries
    """
    df = df.copy()

    # ID列を保持
    ids = df[config.id_col].copy()

    # 削除列リストを構築（ID列 + 設定で指定された削除列）
    drop_cols = [config.id_col] + config.drop_cols
    X = df.drop(columns=drop_cols)

    # カテゴリ変数をcategory型に変換（LightGBM用）
    for col in config.categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
        else:
            print(f"Warning: Categorical column '{col}' not found in features")

    # カテゴリカル列がテストデータに存在することを検証
    missing_cats = [col for col in config.categorical_cols if col not in X.columns]
    if missing_cats:
        raise ValueError(f"テストデータにカテゴリカル列が見つかりません: {missing_cats}")

    return X, ids
