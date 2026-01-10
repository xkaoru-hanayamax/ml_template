# Titanic LightGBM Prediction

Kaggle Titanicデータセットを使用したLightGBM生存予測システム。

## Features

- **シンプルな実装**: LightGBMのネイティブ機能を活用し、欠損値補完・エンコーディング不要
- **K-Fold CV**: 5分割交差検証による精度評価
- **ハイパーパラメータ最適化**: Optunaによる自動チューニング（オプション）
- **Docker対応**: 環境構築不要で実行可能

## Performance

- **CV Accuracy**: 84.06% ± 2.01%
- **Expected Kaggle Score**: 0.76-0.78 (Public LB)

## Quick Start Commands

```bash
# Option 1: Quick validation (hardcoded params, ~10 seconds)
python main.py train

# Option 2: Optimize hyperparameters with Optuna (~10-20 minutes, recommended)
python main.py optimize

# Generate predictions (works with both train and optimize)
python main.py predict
```

## 使い方

### 方法1: Docker

```bash
# イメージビルド
docker build -t titanic-lightgbm .

# 学習実行
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/output:/app/output" \
  titanic-lightgbm python main.py train

# 予測実行
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/output:/app/output" \
  titanic-lightgbm python main.py predict
```

### 方法2: Docker Compose

```bash
# コンテナ起動
docker-compose up -d

# コンテナ内で作業
docker-compose exec titanic bash

# コンテナ内で実行
python main.py train
python main.py predict

# コンテナ停止
docker-compose down
```

### 方法3: ローカル環境

```bash
# 依存パッケージインストール
pip install -r requirements.txt

# 実行
python main.py train
python main.py predict
```

## Workflow Comparison

### Simple Workflow (train.py)
```bash
python main.py train    # K-Fold CV with hardcoded params
python main.py predict  # Generate submission
```
- **Use case**: Quick model validation, baseline results
- **Output**: models/lightgbm_model.txt
- **Time**: ~5-10 seconds

### Optimized Workflow (optimize.py)
```bash
python main.py optimize # Optuna tuning (100 trials) + final model
python main.py predict  # Generate submission
```
- **Use case**: Best model performance for Kaggle submission
- **Output**:
  - models/lightgbm_model.pkl (pickled model)
  - models/lightgbm_params.json (best params + CV results)
  - models/optuna_study.pkl (optional, for analysis)
- **Time**: ~10-20 minutes

## Project Structure

```
project/
├── data/
│   ├── train.csv               # 訓練データ
│   └── test.csv                # テストデータ
├── models/
│   ├── lightgbm_model.txt      # Model from train.py
│   ├── lightgbm_model.pkl      # Model from optimize.py (pickled)
│   ├── lightgbm_params.json    # Best params + CV results from optimize.py
│   └── optuna_study.pkl        # Optuna study object (optional)
├── output/
│   └── submission.csv          # Kaggle提出用ファイル
├── src/
│   ├── preprocessor.py         # 前処理（列削除のみ）
│   ├── train.py                # Quick training (validation mode)
│   ├── optimize.py             # Optuna hyperparameter tuning + final model
│   └── predict.py              # 予測コマンド
├── main.py                     # CLIエントリーポイント
├── Dockerfile                  # Docker設定
├── docker-compose.yml          # Docker Compose設定
└── requirements.txt            # 依存パッケージ
```

## Implementation Details

### 前処理
- **削除列**: PassengerId, Name, Ticket, Cabin
- **使用特徴量**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked (7列)
- **カテゴリ変数**: Sex, Embarked（category型に変換、LightGBMが自動処理）
- **欠損値**: Age, Embarked, Fareの欠損値はLightGBMが自動処理

### モデル
- **アルゴリズム**: LightGBM (Gradient Boosting)
- **パラメータ**:
  - objective: binary
  - learning_rate: 0.05
  - num_leaves: 31
  - feature_fraction: 0.9
- **Early Stopping**: 50ラウンド
- **CV**: Stratified 5-Fold

### ハイパーパラメータ最適化 (optimize.py)

- **Framework**: Optuna
- **Search Space**:
  - num_leaves: [15, 63]
  - learning_rate: [0.01, 0.1] (log scale)
  - feature_fraction: [0.6, 1.0]
  - bagging_fraction: [0.6, 1.0]
  - bagging_freq: [1, 7]
  - min_child_samples: [5, 50]
  - lambda_l1: [0.0, 10.0]
  - lambda_l2: [0.0, 10.0]
- **Trials**: 100 (default)
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Pruner**: Median pruner for early stopping
- **Objective**: Maximize mean accuracy across 5-fold CV

### 特徴量重要度（学習結果）
1. Sex: 2768
2. Age: 1716
3. Fare: 1695
4. Pclass: 967
5. SibSp: 197
6. Embarked: 169
7. Parch: 114

## Output

### models/lightgbm_model.txt
Trained LightGBM model from `train.py` (text format)

### models/lightgbm_model.pkl
Trained LightGBM model from `optimize.py` (pickled format)

### models/lightgbm_params.json
Best hyperparameters and CV results from `optimize.py`:
- best_params: Optimized hyperparameters
- cv_results: Mean/std accuracy, individual fold scores
- best_iteration: Number of boosting rounds
- optuna_study: Study metadata

### output/submission.csv
Kaggle提出用ファイル（PassengerId, Survived列）

## Requirements

- Python 3.11
- pandas 2.1.4
- numpy 1.26.2
- scikit-learn 1.3.2
- lightgbm 4.1.0
- optuna 3.5.0
- joblib 1.3.2

または

- Docker

## License

MIT
