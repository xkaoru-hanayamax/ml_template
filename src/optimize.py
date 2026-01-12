import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pickle
import json
import joblib
import os
from datetime import datetime

try:
    from src.preprocessor import preprocess_train
    from src.config import DatasetConfig
except ImportError:
    # For Jupyter Notebook compatibility
    preprocess_train = None
    DatasetConfig = None


# Global variables for data (avoid re-loading in each trial)
X_global = None
y_global = None
config_global = None
categorical_cols_global = None


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization

    Args:
        trial: Optuna trial object

    Returns:
        float: Mean accuracy across 5-fold CV
    """
    global categorical_cols_global
    # Suggest hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'verbose': -1,
        'random_state': 42
    }

    num_boost_round = 1000
    early_stopping_rounds = 50

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_global, y_global), 1):
        X_train_fold, X_val_fold = X_global.iloc[train_idx], X_global.iloc[val_idx]
        y_train_fold, y_val_fold = y_global.iloc[train_idx], y_global.iloc[val_idx]

        # LightGBM Dataset作成
        train_data = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_cols_global
        )
        val_data = lgb.Dataset(
            X_val_fold,
            label=y_val_fold,
            categorical_feature=categorical_cols_global,
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

        # Report intermediate value for pruning
        trial.report(np.mean(cv_scores[:fold]), fold)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(cv_scores)


def run_optimize(config: DatasetConfig, n_trials: int = 100):
    """
    Run Optuna hyperparameter optimization and train final model

    Args:
        config: データセット設定
        n_trials: Number of Optuna trials (default: 100)
    """
    global X_global, y_global, config_global

    print("=" * 60)
    print("Optuna Hyperparameter Optimization + Final Model Training")
    print("=" * 60)

    # 1. Load and preprocess data
    print("\n[1/6] Loading training data...")
    config_global = config
    train_df = pd.read_csv(config.train_path)
    print(f"  Loaded {len(train_df)} rows")

    X_global, y_global = preprocess_train(train_df, config)
    print(f"  Features: {list(X_global.columns)}")
    print(f"  Shape: {X_global.shape}")
    print(f"  Missing values: {X_global.isnull().sum().to_dict()}")

    # 2. Create Optuna study
    print(f"\n[2/6] Running Optuna optimization ({n_trials} trials)...")
    print("  This may take 10-20 minutes...")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner()
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 3. Display optimization results
    print(f"\n[3/6] Optimization Results:")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best accuracy: {study.best_value:.4f}")
    print(f"  Best parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key:20s}: {value:.6f}")
        else:
            print(f"    {key:20s}: {value}")

    # 4. Train final model with best parameters
    print(f"\n[4/6] Training final model with best parameters...")
    best_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        **study.best_params,
        'verbose': -1,
        'random_state': 42
    }

    num_boost_round = 1000
    early_stopping_rounds = 50

    # Perform final K-Fold CV to get accurate scores with best params
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_global, y_global), 1):
        X_train_fold, X_val_fold = X_global.iloc[train_idx], X_global.iloc[val_idx]
        y_train_fold, y_val_fold = y_global.iloc[train_idx], y_global.iloc[val_idx]

        # LightGBM Dataset作成
        train_data = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_cols_global
        )
        val_data = lgb.Dataset(
            X_val_fold,
            label=y_val_fold,
            categorical_feature=categorical_cols_global,
            reference=train_data
        )

        # 学習
        model = lgb.train(
            best_params,
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

    print(f"\n  Final CV Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"  Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean best iteration: {int(np.mean(best_iterations))}")

    # Train final model on full dataset
    print("\n  Training on full dataset...")
    mean_best_iteration = int(np.mean(best_iterations))
    final_train_data = lgb.Dataset(
        X_global,
        label=y_global,
        categorical_feature=categorical_cols_global
    )

    final_model = lgb.train(
        best_params,
        final_train_data,
        num_boost_round=mean_best_iteration,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # 5. Save outputs
    print(f"\n[5/6] Saving outputs...")
    os.makedirs('models', exist_ok=True)

    # Save model (pkl)
    model_path = 'models/lightgbm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"  Model saved to {model_path}")

    # Save parameters (json)
    params_data = {
        'best_params': best_params,
        'cv_results': {
            'mean_accuracy': float(np.mean(cv_scores)),
            'std_accuracy': float(np.std(cv_scores)),
            'fold_scores': [float(s) for s in cv_scores]
        },
        'best_iteration': mean_best_iteration,
        'optuna_study': {
            'n_trials': len(study.trials),
            'best_trial_number': study.best_trial.number,
            'best_value': float(study.best_value)
        },
        'timestamp': datetime.now().isoformat()
    }

    params_path = 'models/lightgbm_params.json'
    with open(params_path, 'w') as f:
        json.dump(params_data, f, indent=2)
    print(f"  Parameters saved to {params_path}")

    # Save study (optional, for later analysis)
    study_path = 'models/optuna_study.pkl'
    joblib.dump(study, study_path)
    print(f"  Study saved to {study_path}")

    # 6. Feature importance
    print(f"\n[6/6] Feature Importance:")
    importance = final_model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': X_global.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:12s}: {row['importance']:8.0f}")

    print("\n" + "=" * 60)
    print("Optimization completed successfully!")
    print("=" * 60)
    print(f"\nNext step: Run 'python main.py predict' to generate predictions")


# ========== Jupyter Notebook用の独立した関数 ==========

def optimize_hyperparameters(
    train_data_path: str,
    target_col: str,
    id_col: str,
    categorical_cols: list = None,
    n_trials: int = 100,
    n_folds: int = 5,
    model_output_dir: str = 'models'
) -> dict:
    """
    Optunaでハイパーパラメータ最適化を実行し、最終モデルを学習（Jupyter Notebook用）

    この関数は完全に独立しており、src モジュールへの依存はありません。

    Args:
        train_data_path: 前処理済み訓練データのCSVパス（例: 'processed_data/processed_train.csv'）
        target_col: ターゲット列の名前（例: 'Survived'）
        id_col: ID列の名前（例: 'PassengerId'）
        categorical_cols: カテゴリカル列のリスト（例: ['Sex', 'Embarked']）
        n_trials: Optunaの試行回数（デフォルト: 100）
        n_folds: CVのフォールド数（デフォルト: 5）
        model_output_dir: モデル保存ディレクトリ（デフォルト: 'models'）

    Returns:
        dict: 以下のキーを含む辞書
            - 'model_path': 保存されたモデルのパス（pkl形式）
            - 'params_path': ベストパラメータのJSONパス
            - 'study_path': Optunaスタディのpklパス
            - 'best_params': ベストパラメータの辞書
            - 'best_score': ベストスコア

    Jupyter使用例:
        ```python
        result = optimize_hyperparameters(
            train_data_path='processed_data/processed_train.csv',
            target_col='Survived',
            id_col='PassengerId',
            categorical_cols=['Sex', 'Embarked'],
            n_trials=50
        )
        print(f"Best score: {result['best_score']:.4f}")
        print(f"Model saved to: {result['model_path']}")
        ```
    """
    global X_global, y_global, categorical_cols_global

    if categorical_cols is None:
        categorical_cols = []

    print("=" * 60)
    print("Optuna Hyperparameter Optimization (Jupyter Notebook)")
    print("=" * 60)

    # 1. データ読込と前処理
    print(f"\n[1/6] Loading training data from {train_data_path}...")
    df = pd.read_csv(train_data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # カテゴリカル列を'category'型に変換
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            print(f"  Warning: Categorical column '{col}' not found in data")

    # ターゲットとIDを分離
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in data")

    y_global = df[target_col]
    X_global = df.drop(columns=[id_col, target_col])
    categorical_cols_global = categorical_cols

    print(f"  Features: {list(X_global.columns)}")
    print(f"  Shape: {X_global.shape}")
    print(f"  Missing values: {X_global.isnull().sum().to_dict()}")

    # 2. Optunaスタディの作成と最適化
    print(f"\n[2/6] Running Optuna optimization ({n_trials} trials)...")
    print("  This may take 10-20 minutes...")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner()
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 3. 最適化結果の表示
    print(f"\n[3/6] Optimization Results:")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best accuracy: {study.best_value:.4f}")
    print(f"  Best parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key:20s}: {value:.6f}")
        else:
            print(f"    {key:20s}: {value}")

    # 4. ベストパラメータで最終モデルを学習
    print(f"\n[4/6] Training final model with best parameters...")
    best_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        **study.best_params,
        'verbose': -1,
        'random_state': 42
    }

    num_boost_round = 1000
    early_stopping_rounds = 50

    # K-Fold CV で最終スコアと最適イテレーション数を取得
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_global, y_global), 1):
        X_train_fold, X_val_fold = X_global.iloc[train_idx], X_global.iloc[val_idx]
        y_train_fold, y_val_fold = y_global.iloc[train_idx], y_global.iloc[val_idx]

        # LightGBM Dataset作成
        train_data = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_cols
        )
        val_data = lgb.Dataset(
            X_val_fold,
            label=y_val_fold,
            categorical_feature=categorical_cols,
            reference=train_data
        )

        # 学習
        model = lgb.train(
            best_params,
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

    print(f"\n  Final CV Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"  Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean best iteration: {int(np.mean(best_iterations))}")

    # 全データで最終モデルを学習
    print("\n  Training on full dataset...")
    mean_best_iteration = int(np.mean(best_iterations))
    final_train_data = lgb.Dataset(
        X_global,
        label=y_global,
        categorical_feature=categorical_cols
    )

    final_model = lgb.train(
        best_params,
        final_train_data,
        num_boost_round=mean_best_iteration,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # 5. 出力の保存
    print(f"\n[5/6] Saving outputs...")
    os.makedirs(model_output_dir, exist_ok=True)

    # モデル保存（pkl）
    model_path = os.path.join(model_output_dir, 'lightgbm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"  Model saved to {model_path}")

    # パラメータ保存（JSON）
    params_data = {
        'best_params': best_params,
        'cv_results': {
            'mean_accuracy': float(np.mean(cv_scores)),
            'std_accuracy': float(np.std(cv_scores)),
            'fold_scores': [float(s) for s in cv_scores]
        },
        'best_iteration': mean_best_iteration,
        'optuna_study': {
            'n_trials': len(study.trials),
            'best_trial_number': study.best_trial.number,
            'best_value': float(study.best_value)
        },
        'timestamp': datetime.now().isoformat()
    }

    params_path = os.path.join(model_output_dir, 'lightgbm_params.json')
    with open(params_path, 'w') as f:
        json.dump(params_data, f, indent=2)
    print(f"  Parameters saved to {params_path}")

    # スタディ保存（optional、後で分析可能）
    study_path = os.path.join(model_output_dir, 'optuna_study.pkl')
    joblib.dump(study, study_path)
    print(f"  Study saved to {study_path}")

    # 6. 特徴量重要度
    print(f"\n[6/6] Feature Importance:")
    importance = final_model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': X_global.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:12s}: {row['importance']:8.0f}")

    print("\n" + "=" * 60)
    print("Optimization completed successfully!")
    print("=" * 60)

    return {
        'model_path': model_path,
        'params_path': params_path,
        'study_path': study_path,
        'best_params': best_params,
        'best_score': float(study.best_value)
    }


if __name__ == '__main__':
    run_optimize()
