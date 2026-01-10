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

from src.preprocessor import preprocess_train


# Global variables for data (avoid re-loading in each trial)
X_global = None
y_global = None
categorical_features = ['Sex', 'Embarked']


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization

    Args:
        trial: Optuna trial object

    Returns:
        float: Mean accuracy across 5-fold CV
    """
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

        # Report intermediate value for pruning
        trial.report(np.mean(cv_scores[:fold]), fold)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(cv_scores)


def run_optimize(n_trials=100):
    """
    Run Optuna hyperparameter optimization and train final model

    - Optimize hyperparameters with Optuna (100 trials by default)
    - Train final model with best parameters on full dataset
    - Save model as pkl file
    - Save parameters and CV results as json file

    Args:
        n_trials: Number of Optuna trials (default: 100)
    """
    global X_global, y_global

    print("=" * 60)
    print("Optuna Hyperparameter Optimization + Final Model Training")
    print("=" * 60)

    # 1. Load and preprocess data
    print("\n[1/6] Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    print(f"  Loaded {len(train_df)} rows")

    X_global, y_global = preprocess_train(train_df)
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
        categorical_feature=categorical_features
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


if __name__ == '__main__':
    run_optimize()
