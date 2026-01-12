import pandas as pd
import json
import os
from typing import Tuple

try:
    from src.config import DatasetConfig
except ImportError:
    # For Jupyter Notebook compatibility
    DatasetConfig = None


def preprocess_train(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.Series]:
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


def preprocess_test(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.Series]:
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


# ========== Jupyter Notebook用の独立した関数 ==========

def load_metadata(metadata_path: str) -> dict:
    """
    メタデータJSONファイルを読み込む

    Args:
        metadata_path: メタデータJSONファイルのパス

    Returns:
        メタデータの辞書

    Example:
        metadata = load_metadata('processed_data/metadata.json')
        categorical_cols = metadata['categorical_cols']
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_data(
    train_path: str,
    test_path: str,
    target_col: str,
    id_col: str,
    drop_cols: list,
    categorical_cols: list = None,
    output_dir: str = 'processed_data'
) -> dict:
    """
    訓練・テストデータを前処理してCSVファイルに保存する（Jupyter Notebook用）

    この関数は完全に独立しており、src モジュールへの依存はありません。

    Args:
        train_path: 訓練データのCSVファイルパス（例: 'data/train.csv'）
        test_path: テストデータのCSVファイルパス（例: 'data/test.csv'）
        target_col: ターゲット列の名前（例: 'Survived'）
        id_col: ID列の名前（例: 'PassengerId'）
        drop_cols: 削除する列名のリスト（例: ['Name', 'Ticket', 'Cabin']）
        categorical_cols: カテゴリカル列のリスト（例: ['Sex', 'Embarked']）
        output_dir: 出力ディレクトリ（デフォルト: 'processed_data'）

    Returns:
        dict: 以下のキーを含む辞書
            - 'train_output': 前処理済み訓練データのパス
            - 'test_output': 前処理済みテストデータのパス
            - 'metadata': メタデータJSONのパス

    Jupyter使用例:
        ```python
        result = preprocess_data(
            train_path='data/train.csv',
            test_path='data/test.csv',
            target_col='Survived',
            id_col='PassengerId',
            drop_cols=['Name', 'Ticket', 'Cabin'],
            categorical_cols=['Sex', 'Embarked']
        )
        print(f"Processed data saved to: {result['train_output']}")
        ```
    """
    if categorical_cols is None:
        categorical_cols = []

    print("=" * 60)
    print("Preprocessing Data for Jupyter Notebook")
    print("=" * 60)

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 1. 訓練データの読み込み
    print(f"\n[1/4] Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"  Loaded {len(train_df)} rows, {len(train_df.columns)} columns")

    # 2. 訓練データの前処理
    print("\n[2/4] Preprocessing training data...")

    # drop_cols に含まれる列を削除（ID列とターゲット列は保持）
    cols_to_drop = [col for col in drop_cols if col in train_df.columns and col != id_col and col != target_col]
    train_processed = train_df.drop(columns=cols_to_drop)

    # カテゴリカル列を文字列型に変換（CSV保存のため）
    for col in categorical_cols:
        if col in train_processed.columns:
            train_processed[col] = train_processed[col].astype(str)

    # 特徴量の列名を取得（ID列とターゲット列を除く）
    feature_names = [col for col in train_processed.columns if col != id_col and col != target_col]

    print(f"  Original columns: {list(train_df.columns)}")
    print(f"  Dropped columns: {cols_to_drop}")
    print(f"  Feature columns: {feature_names}")
    print(f"  Target column: {target_col}")
    print(f"  ID column: {id_col}")
    print(f"  Categorical columns: {categorical_cols}")

    # 訓練データを保存
    train_output_path = os.path.join(output_dir, 'processed_train.csv')
    train_processed.to_csv(train_output_path, index=False)
    print(f"  Saved to: {train_output_path}")

    # 3. テストデータの読み込み
    print(f"\n[3/4] Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    print(f"  Loaded {len(test_df)} rows, {len(test_df.columns)} columns")

    # 4. テストデータの前処理
    print("\n[4/4] Preprocessing test data...")

    # drop_cols に含まれる列を削除（ID列は保持、ターゲット列はテストデータには存在しない）
    cols_to_drop_test = [col for col in drop_cols if col in test_df.columns and col != id_col]
    test_processed = test_df.drop(columns=cols_to_drop_test)

    # カテゴリカル列を文字列型に変換（CSV保存のため）
    for col in categorical_cols:
        if col in test_processed.columns:
            test_processed[col] = test_processed[col].astype(str)
        else:
            print(f"  Warning: Categorical column '{col}' not found in test data")

    # テストデータの特徴量を検証
    test_feature_names = [col for col in test_processed.columns if col != id_col]
    print(f"  Original columns: {list(test_df.columns)}")
    print(f"  Dropped columns: {cols_to_drop_test}")
    print(f"  Feature columns: {test_feature_names}")
    print(f"  ID column: {id_col}")

    # テストデータを保存
    test_output_path = os.path.join(output_dir, 'processed_test.csv')
    test_processed.to_csv(test_output_path, index=False)
    print(f"  Saved to: {test_output_path}")

    # 5. メタデータの保存
    print("\n[5/5] Saving metadata...")
    metadata = {
        'categorical_cols': categorical_cols,
        'feature_names': feature_names,
        'target_col': target_col,
        'id_col': id_col,
        'drop_cols': drop_cols
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)

    return {
        'train_output': train_output_path,
        'test_output': test_output_path,
        'metadata': metadata_path
    }
