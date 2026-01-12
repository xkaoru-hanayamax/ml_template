import pandas as pd
import lightgbm as lgb
import pickle
import os

try:
    from src.preprocessor import preprocess_test
    from src.config import DatasetConfig
except ImportError:
    # For Jupyter Notebook compatibility
    preprocess_test = None
    DatasetConfig = None


def run_predict(config: DatasetConfig):
    """
    学習済みモデルで予測を実行

    Args:
        config: データセット設定
    """
    print("=" * 60)
    print("Generating Predictions")
    print("=" * 60)

    # 1. モデル存在確認（pkl優先、txtにフォールバック）
    model_pkl_path = 'models/lightgbm_model.pkl'
    model_txt_path = 'models/lightgbm_model.txt'

    if os.path.exists(model_pkl_path):
        model_path = model_pkl_path
        load_method = 'pkl'
    elif os.path.exists(model_txt_path):
        model_path = model_txt_path
        load_method = 'txt'
    else:
        print(f"\nError: No model found")
        print("Please run: python main.py train OR python main.py optimize")
        return

    # 2. データ読込
    print("\n[1/5] Loading test data...")
    test_df = pd.read_csv(config.test_path)
    print(f"  Loaded {len(test_df)} rows")

    # 3. 前処理
    print("\n[2/5] Preprocessing...")
    X_test, ids = preprocess_test(test_df, config)
    print(f"  Features: {list(X_test.columns)}")
    print(f"  Shape: {X_test.shape}")
    print(f"  Missing values: {X_test.isnull().sum().to_dict()}")

    # 4. モデルロード
    print(f"\n[3/5] Loading model from {model_path}...")
    if load_method == 'pkl':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = lgb.Booster(model_file=model_path)
    print("  Model loaded successfully")

    # 5. 予測
    print("\n[4/5] Generating predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print(f"  Predicted survival rate: {y_pred.mean():.2%}")
    print(f"  Distribution: {pd.Series(y_pred).value_counts().to_dict()}")

    # 6. Submission作成
    print("\n[5/5] Creating submission file...")
    submission = pd.DataFrame({
        config.id_col: ids,
        config.submission_col_target: y_pred
    })

    os.makedirs('output', exist_ok=True)
    output_path = 'output/submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"  Submission saved to {output_path}")

    # サンプル表示
    print("\n  Sample predictions:")
    print(submission.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Prediction completed successfully!")
    print("=" * 60)


# ========== Jupyter Notebook用の独立した関数 ==========

def predict(
    test_data_path: str,
    model_path: str,
    id_col: str,
    target_col: str,
    categorical_cols: list = None,
    output_path: str = 'output/submission.csv'
) -> pd.DataFrame:
    """
    学習済みモデルで予測を実行（Jupyter Notebook用）

    この関数は完全に独立しており、src モジュールへの依存はありません。

    Args:
        test_data_path: 前処理済みテストデータのCSVパス（例: 'processed_data/processed_test.csv'）
        model_path: 学習済みモデルのパス（例: 'models/lightgbm_model.txt' or 'models/lightgbm_model.pkl'）
        id_col: ID列の名前（例: 'PassengerId'）
        target_col: ターゲット列の名前（例: 'Survived'）
        categorical_cols: カテゴリカル列のリスト（例: ['Sex', 'Embarked']）
        output_path: 出力CSVのパス（デフォルト: 'output/submission.csv'）

    Returns:
        pd.DataFrame: ID列と予測結果を含むDataFrame

    Jupyter使用例:
        ```python
        submission = predict(
            test_data_path='processed_data/processed_test.csv',
            model_path='models/lightgbm_model.txt',
            id_col='PassengerId',
            target_col='Survived',
            categorical_cols=['Sex', 'Embarked']
        )
        print(submission.head())
        ```
    """
    if categorical_cols is None:
        categorical_cols = []

    print("=" * 60)
    print("Generating Predictions (Jupyter Notebook)")
    print("=" * 60)

    # 1. モデル存在確認
    print(f"\n[1/5] Checking model at {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # モデルフォーマットを判定
    if model_path.endswith('.pkl'):
        load_method = 'pkl'
        print("  Model format: pickle")
    elif model_path.endswith('.txt'):
        load_method = 'txt'
        print("  Model format: text")
    else:
        raise ValueError(f"Unsupported model format. Expected .pkl or .txt, got: {model_path}")

    # 2. データ読込
    print(f"\n[2/5] Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # カテゴリカル列を'category'型に変換
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            print(f"  Warning: Categorical column '{col}' not found in data")

    # IDを分離
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in data")

    ids = df[id_col]
    X_test = df.drop(columns=[id_col])

    print(f"  Features: {list(X_test.columns)}")
    print(f"  Shape: {X_test.shape}")
    print(f"  Missing values: {X_test.isnull().sum().to_dict()}")

    # 3. モデルロード
    print(f"\n[3/5] Loading model from {model_path}...")
    if load_method == 'pkl':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = lgb.Booster(model_file=model_path)
    print("  Model loaded successfully")

    # 4. 予測
    print("\n[4/5] Generating predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print(f"  Predicted survival rate: {y_pred.mean():.2%}")
    print(f"  Distribution: {pd.Series(y_pred).value_counts().to_dict()}")

    # 5. Submission作成
    print(f"\n[5/5] Creating submission file...")
    submission = pd.DataFrame({
        id_col: ids,
        target_col: y_pred
    })

    # 出力ディレクトリの作成
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"  Submission saved to {output_path}")

    # サンプル表示
    print("\n  Sample predictions:")
    print(submission.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Prediction completed successfully!")
    print("=" * 60)

    return submission


if __name__ == '__main__':
    run_predict()
