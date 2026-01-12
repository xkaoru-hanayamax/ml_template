import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from typing import Optional
import os
import matplotlib
matplotlib.use('Agg')  # 非対話型バックエンド（サーバー環境対応）
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessor import preprocess_train
from src.config import DatasetConfig


def plot_learning_curves(fold_metrics_history, output_dir='output'):
    """
    5-fold CVの学習曲線を平均化してプロット

    Args:
        fold_metrics_history: 各foldのevals_result辞書のリスト
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # 各foldのメトリクスを抽出
    train_losses = []
    valid_losses = []

    for evals_result in fold_metrics_history:
        train_losses.append(evals_result['train']['binary_logloss'])
        valid_losses.append(evals_result['valid']['binary_logloss'])

    # 最大イテレーション数を取得（early stoppingでfoldごとに異なる）
    max_iter = max(len(losses) for losses in train_losses)

    # 短いfoldは最後の値で埋める（early stopping後は変化しないと仮定）
    train_losses_padded = []
    valid_losses_padded = []

    for train_loss, valid_loss in zip(train_losses, valid_losses):
        if len(train_loss) < max_iter:
            padding = [train_loss[-1]] * (max_iter - len(train_loss))
            train_losses_padded.append(train_loss + padding)
            valid_losses_padded.append(valid_loss + padding)
        else:
            train_losses_padded.append(train_loss)
            valid_losses_padded.append(valid_loss)

    # numpy配列に変換して平均・標準偏差を計算
    train_losses_arr = np.array(train_losses_padded)
    valid_losses_arr = np.array(valid_losses_padded)

    train_mean = train_losses_arr.mean(axis=0)
    train_std = train_losses_arr.std(axis=0)
    valid_mean = valid_losses_arr.mean(axis=0)
    valid_std = valid_losses_arr.std(axis=0)

    iterations = np.arange(1, max_iter + 1)

    # プロット作成
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train Loss
    axes[0].plot(iterations, train_mean, label='Mean Train Loss', color='#1f77b4', linewidth=2)
    axes[0].fill_between(iterations, train_mean - train_std, train_mean + train_std,
                         alpha=0.2, color='#1f77b4', label='±1 Std')
    axes[0].set_xlabel('Boosting Iteration', fontsize=12)
    axes[0].set_ylabel('Binary Log Loss', fontsize=12)
    axes[0].set_title('Training Loss (Averaged across 5 folds)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Validation Loss
    axes[1].plot(iterations, valid_mean, label='Mean Valid Loss', color='#ff7f0e', linewidth=2)
    axes[1].fill_between(iterations, valid_mean - valid_std, valid_mean + valid_std,
                         alpha=0.2, color='#ff7f0e', label='±1 Std')
    axes[1].set_xlabel('Boosting Iteration', fontsize=12)
    axes[1].set_ylabel('Binary Log Loss', fontsize=12)
    axes[1].set_title('Validation Loss (Averaged across 5 folds)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Learning curves saved to {save_path}")


def plot_feature_importance(importance_df, output_dir='output'):
    """
    特徴量重要度を可視化

    Args:
        importance_df: feature/importanceカラムを持つDataFrame
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    # 重要度降順でソート済みのDataFrameを使用
    sns.barplot(data=importance_df, x='importance', y='feature',
                hue='feature', palette='viridis', legend=False, ax=ax)
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Feature importance plot saved to {save_path}")


def plot_partial_dependence(model, X, categorical_features, output_dir='output'):
    """
    全特徴量のPartial Dependence Plotを手動で生成（LightGBM対応）

    Args:
        model: 学習済みLightGBMモデル
        X: 特徴量DataFrame
        categorical_features: カテゴリ変数のリスト
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # 特徴量名を取得
    feature_names = list(X.columns)

    # PDPを生成（3x3グリッド）
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # 各特徴量に対して部分依存を計算
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]

        if feature in categorical_features:
            # カテゴリ変数の場合
            categories = X[feature].cat.categories
            pdp_values = []

            for category in categories:
                # 全サンプルのコピーを作成し、対象特徴量だけをこのカテゴリに固定
                X_temp = X.copy()
                X_temp[feature] = pd.Categorical([category] * len(X), categories=categories)
                # 予測値の平均を計算
                pred = model.predict(X_temp)
                pdp_values.append(pred.mean())

            # バーチャートでプロット
            ax.bar(range(len(categories)), pdp_values, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=0)
            ax.set_ylabel('Partial Dependence', fontsize=10)
            ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        else:
            # 数値変数の場合
            # グリッド値を生成（パーセンタイルベース）
            grid_values = np.linspace(
                X[feature].quantile(0.05),
                X[feature].quantile(0.95),
                50
            )
            pdp_values = []

            for grid_value in grid_values:
                # 全サンプルのコピーを作成し、対象特徴量だけをこの値に固定
                X_temp = X.copy()
                X_temp[feature] = grid_value
                # 予測値の平均を計算
                pred = model.predict(X_temp)
                pdp_values.append(pred.mean())

            # 折れ線グラフでプロット
            ax.plot(grid_values, pdp_values, color='steelblue', linewidth=2)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Partial Dependence', fontsize=10)
            ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

    # 使用していない8番目と9番目のサブプロットを非表示
    axes[7].set_visible(False)
    axes[8].set_visible(False)

    fig.suptitle('Partial Dependence Plots (All Features)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'partial_dependence_plots.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Partial dependence plots saved to {save_path}")


def run_train(config: DatasetConfig,
              params: Optional[dict] = None,
              num_boost_round: int = 1000,
              early_stopping_rounds: int = 50):
    """
    LightGBMモデルの学習を実行

    Args:
        config: データセット設定
        params: LightGBMパラメータ（Noneの場合はデフォルト値を使用）
        num_boost_round: ブースティング回数
        early_stopping_rounds: Early stoppingのラウンド数
    """
    print("=" * 60)
    print("Training LightGBM Model")
    print("=" * 60)

    # 1. データ読込
    print("\n[1/6] Loading training data...")
    train_df = pd.read_csv(config.train_path)
    print(f"  Loaded {len(train_df)} rows")

    # 2. 前処理
    print("\n[2/6] Preprocessing...")
    X, y = preprocess_train(train_df, config)
    print(f"  Features: {list(X.columns)}")
    print(f"  Shape: {X.shape}")
    print(f"  Missing values: {X.isnull().sum().to_dict()}")

    # 3. LightGBMパラメータ設定
    if params is None:
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

    categorical_features = config.categorical_cols

    # 4. K-Fold Cross Validation
    print("\n[3/7] Running 5-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_iterations = []
    fold_metrics_history = []  # 各foldのメトリクスを保存

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

        # メトリクスを記録するための辞書
        evals_result = {}

        # 学習
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result)  # メトリクスを記録
            ]
        )

        # このfoldのメトリクスを保存
        fold_metrics_history.append(evals_result)

        # 検証データで予測
        y_pred_proba = model.predict(X_val_fold, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(accuracy)
        best_iterations.append(model.best_iteration)

        print(f"  Fold {fold}: Accuracy = {accuracy:.4f}, Best iteration = {model.best_iteration}")

    # 5. CV結果サマリ
    print("\n[4/7] Cross Validation Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"  Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean best iteration: {int(np.mean(best_iterations))}")

    # 5a. 学習曲線の生成
    print("\n[5/7] Generating learning curves...")
    plot_learning_curves(fold_metrics_history, output_dir='output')

    # 6. 全訓練データで最終モデル学習
    print("\n[6/7] Training final model on full dataset...")
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
    print("\n[7/7] Feature Importance:")
    importance = final_model.feature_importance(importance_type='gain')
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:12s}: {row['importance']:8.0f}")

    # 8a. 特徴量重要度の可視化
    print("\n  Generating feature importance plot...")
    plot_feature_importance(importance_df, output_dir='output')

    # 8b. 部分依存プロットの生成
    print("\n  Generating partial dependence plots...")
    plot_partial_dependence(final_model, X, config.categorical_cols, output_dir='output')

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    run_train()
