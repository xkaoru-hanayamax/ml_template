import pandas as pd
import lightgbm as lgb
import pickle
import os

from src.preprocessor import preprocess_test
from src.config import DatasetConfig


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


if __name__ == '__main__':
    run_predict()
