import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

from src.preprocessor import preprocess_train


def run_train():
    """
    LightGBMモデルの学習を実行
    - K-Fold Cross Validationで評価
    - 全訓練データで最終モデルを学習
    - モデルをmodels/lightgbm_model.txtに保存
    """
    print("=" * 60)
    print("Training LightGBM Model")
    print("=" * 60)

    # 1. データ読込
    print("\n[1/6] Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    print(f"  Loaded {len(train_df)} rows")

    # 2. 前処理
    print("\n[2/6] Preprocessing...")
    X, y = preprocess_train(train_df)
    print(f"  Features: {list(X.columns)}")
    print(f"  Shape: {X.shape}")
    print(f"  Missing values: {X.isnull().sum().to_dict()}")

    # 3. LightGBMパラメータ設定
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
        'random_state': 42
    }
    num_boost_round = 1000
    early_stopping_rounds = 50
    categorical_features = ['Sex', 'Embarked']

    # 4. K-Fold Cross Validation
    print("\n[3/6] Running 5-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # LightGBM Dataset作成
        train_data = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_features
        )
        val_data = lgb.Dataset(
            X_val_fold,
            label=y_val_fold,
            categorical_feature=categorical_features,
            reference=train_data
        )

        # 学習
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0)
            ]
        )

        # 検証データで予測
        y_pred_proba = model.predict(X_val_fold, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(accuracy)
        best_iterations.append(model.best_iteration)

        print(f"  Fold {fold}: Accuracy = {accuracy:.4f}, Best iteration = {model.best_iteration}")

    # 5. CV結果サマリ
    print("\n[4/6] Cross Validation Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"  Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean best iteration: {int(np.mean(best_iterations))}")

    # 6. 全訓練データで最終モデル学習
    print("\n[5/6] Training final model on full dataset...")
    final_train_data = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_features
    )

    final_model = lgb.train(
        params,
        final_train_data,
        num_boost_round=int(np.mean(best_iterations)),
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # 7. モデル保存
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lightgbm_model.txt'
    final_model.save_model(model_path)
    print(f"  Model saved to {model_path}")

    # 8. 特徴量重要度
    print("\n[6/6] Feature Importance:")
    importance = final_model.feature_importance(importance_type='gain')
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:12s}: {row['importance']:8.0f}")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    run_train()
