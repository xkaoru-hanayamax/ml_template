# Titanic LightGBM Prediction

Kaggle Titanicデータセットを使用したLightGBM生存予測システム。
**CLI（コマンドライン）とJupyter Notebookの両方に対応。**

## 🚀 Features

- **2つの使用方法**: CLI（main.py）とJupyter Notebook（独立関数）の両対応
- **モジュール独立性**: 各モジュールは完全に独立し、Jupyter Notebookでコピペ実行可能
- **シンプルな実装**: LightGBMのネイティブ機能を活用し、欠損値補完・エンコーディング不要
- **K-Fold CV**: 5分割交差検証による精度評価
- **ハイパーパラメータ最適化**: Optunaによる自動チューニング
- **可視化**: 学習曲線、特徴量重要度、部分依存プロットを自動生成
- **Docker対応**: 環境構築不要で実行可能

## 📊 Performance

- **CV Accuracy**: 83.39% ± 0.21% (2-Fold), 84.06% ± 2.01% (5-Fold with optimization)
- **Expected Kaggle Score**: 0.76-0.78 (Public LB)

---

## 🎯 Quick Start

### 方法1: Jupyter Notebook（推奨）

各モジュールを独立して実行でき、セル単位で実験可能です。

```python
# Cell 1: セットアップ
import sys
sys.path.insert(0, 'src')

# Cell 2: データの前処理
from preprocessor import preprocess_data

result = preprocess_data(
    train_path='data/train.csv',
    test_path='data/test.csv',
    target_col='Survived',
    id_col='PassengerId',
    drop_cols=['Name', 'Ticket', 'Cabin'],
    categorical_cols=['Sex', 'Embarked']
)

# Cell 3: モデルの学習
from train import train_model

train_result = train_model(
    train_data_path='processed_data/processed_train.csv',
    target_col='Survived',
    id_col='PassengerId',
    categorical_cols=['Sex', 'Embarked'],
    generate_plots=True
)
print(f"Mean accuracy: {train_result['mean_accuracy']:.4f}")

# Cell 4: 予測
from predict import predict

submission = predict(
    test_data_path='processed_data/processed_test.csv',
    model_path='models/lightgbm_model.txt',
    id_col='PassengerId',
    target_col='Survived',
    categorical_cols=['Sex', 'Embarked']
)
print(submission.head())
```

📝 **詳細**: `jupyter_example.py` に完全な使用例を記載

### 方法2: CLI（コマンドライン）

従来通りのコマンドライン実行も可能です。

```bash
# 1. 標準訓練（固定パラメータ、約10秒）
python main.py train --train-data data/train.csv --test-data data/test.csv \
  --target-col Survived --id-col PassengerId \
  --drop-cols Name Ticket Cabin --categorical-cols Sex Embarked

# 2. Optunaによる最適化（約10-20分、推奨）
python main.py optimize --train-data data/train.csv --test-data data/test.csv \
  --target-col Survived --id-col PassengerId \
  --drop-cols Name Ticket Cabin --categorical-cols Sex Embarked

# 3. 予測生成
python main.py predict --train-data data/train.csv --test-data data/test.csv \
  --target-col Survived --id-col PassengerId \
  --drop-cols Name Ticket Cabin --categorical-cols Sex Embarked

# 4. 全パイプライン実行（train → predict）
python main.py all --train-data data/train.csv --test-data data/test.csv \
  --target-col Survived --id-col PassengerId \
  --drop-cols Name Ticket Cabin --categorical-cols Sex Embarked
```

---

## 📁 Project Structure

```
project/
├── data/
│   ├── train.csv                      # 訓練データ（Kaggle提供）
│   └── test.csv                       # テストデータ（Kaggle提供）
│
├── processed_data/                    # 前処理済みデータ（Jupyter使用時）
│   ├── processed_train.csv            # 特徴量 + ターゲット + ID
│   ├── processed_test.csv             # 特徴量 + ID
│   └── metadata.json                  # カテゴリカル列等のメタデータ
│
├── models/
│   ├── lightgbm_model.txt             # train()から生成（テキスト形式）
│   ├── lightgbm_model.pkl             # optimize()から生成（pickle形式）
│   ├── lightgbm_params.json           # ベストパラメータ + CV結果
│   └── optuna_study.pkl               # Optunaスタディ（分析用）
│
├── output/
│   ├── submission.csv                 # Kaggle提出用ファイル
│   ├── learning_curves.png            # 学習曲線（訓練/検証損失）
│   ├── feature_importance.png         # 特徴量重要度
│   └── partial_dependence_plots.png   # 部分依存プロット
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # データセット設定（CLI用）
│   ├── preprocessor.py                # 前処理
│   │   ├── preprocess_data()          # 🆕 Jupyter用独立関数
│   │   └── load_metadata()            # 🆕 メタデータ読込
│   ├── train.py                       # 訓練
│   │   └── train_model()              # 🆕 Jupyter用独立関数
│   ├── optimize.py                    # Optuna最適化
│   │   └── optimize_hyperparameters() # 🆕 Jupyter用独立関数
│   └── predict.py                     # 予測
│       └── predict()                  # 🆕 Jupyter用独立関数
│
├── main.py                            # CLIエントリーポイント
├── jupyter_example.py                 # 🆕 Jupyter使用例
├── test_jupyter_functions.py          # 🆕 独立関数のテスト
│
├── Dockerfile                         # Docker設定
├── docker-compose.yml                 # Docker Compose設定
├── requirements.txt                   # 依存パッケージ
└── README.md                          # このファイル
```

---

## 🔧 Installation

### ローカル環境

```bash
# 依存パッケージのインストール
pip install -r requirements.txt
```

### Docker

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

### Docker Compose

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

---

## 📖 Usage Details

### Jupyter Notebook用の独立関数

各モジュールは `src.*` への依存なしで完全に独立しています。

#### 1. preprocess_data() - データ前処理

```python
from preprocessor import preprocess_data, load_metadata

result = preprocess_data(
    train_path='data/train.csv',
    test_path='data/test.csv',
    target_col='Survived',
    id_col='PassengerId',
    drop_cols=['Name', 'Ticket', 'Cabin'],
    categorical_cols=['Sex', 'Embarked'],
    output_dir='processed_data'
)

# 戻り値
# {
#     'train_output': 'processed_data/processed_train.csv',
#     'test_output': 'processed_data/processed_test.csv',
#     'metadata': 'processed_data/metadata.json'
# }

# メタデータの読み込み（後で使用する場合）
metadata = load_metadata('processed_data/metadata.json')
```

**出力:**
- `processed_train.csv`: 特徴量 + ターゲット + ID
- `processed_test.csv`: 特徴量 + ID
- `metadata.json`: カテゴリカル列等のメタ情報

#### 2. train_model() - モデル訓練

```python
from train import train_model

result = train_model(
    train_data_path='processed_data/processed_train.csv',
    target_col='Survived',
    id_col='PassengerId',
    categorical_cols=['Sex', 'Embarked'],
    params=None,                   # Noneの場合はデフォルトパラメータ
    num_boost_round=1000,
    early_stopping_rounds=50,
    n_folds=5,
    generate_plots=True,           # グラフ生成のON/OFF
    model_output_dir='models',
    plots_output_dir='output'
)

# 戻り値
# {
#     'model_path': 'models/lightgbm_model.txt',
#     'cv_scores': [0.8318, 0.8360, ...],
#     'mean_accuracy': 0.8339,
#     'std_accuracy': 0.0021,
#     'feature_importance': DataFrame
# }
```

**出力:**
- `models/lightgbm_model.txt`: 訓練済みモデル
- `output/learning_curves.png`: 学習曲線（generate_plots=True時）
- `output/feature_importance.png`: 特徴量重要度
- `output/partial_dependence_plots.png`: 部分依存プロット

#### 3. optimize_hyperparameters() - ハイパーパラメータ最適化

```python
from optimize import optimize_hyperparameters

result = optimize_hyperparameters(
    train_data_path='processed_data/processed_train.csv',
    target_col='Survived',
    id_col='PassengerId',
    categorical_cols=['Sex', 'Embarked'],
    n_trials=100,                  # Optuna試行回数
    n_folds=5,
    model_output_dir='models'
)

# 戻り値
# {
#     'model_path': 'models/lightgbm_model.pkl',
#     'params_path': 'models/lightgbm_params.json',
#     'study_path': 'models/optuna_study.pkl',
#     'best_params': {...},
#     'best_score': 0.8406
# }
```

**出力:**
- `models/lightgbm_model.pkl`: 最適化されたモデル
- `models/lightgbm_params.json`: ベストパラメータ + CV結果
- `models/optuna_study.pkl`: Optunaスタディ（分析用）

#### 4. predict() - 予測生成

```python
from predict import predict

submission = predict(
    test_data_path='processed_data/processed_test.csv',
    model_path='models/lightgbm_model.txt',  # または .pkl
    id_col='PassengerId',
    target_col='Survived',
    categorical_cols=['Sex', 'Embarked'],
    output_path='output/submission.csv'
)

# 戻り値: DataFrameが返される
#    PassengerId  Survived
# 0          892         0
# 1          893         0
# ...
```

**出力:**
- `output/submission.csv`: Kaggle提出用ファイル

### CLI用コマンド一覧

```bash
# 標準訓練（固定パラメータ）
python main.py train --train-data <PATH> --test-data <PATH> \
  --target-col <COL> --id-col <COL> \
  --drop-cols <COL1> <COL2> ... \
  --categorical-cols <COL1> <COL2> ...

# Optuna最適化
python main.py optimize --train-data <PATH> --test-data <PATH> \
  --target-col <COL> --id-col <COL> \
  --drop-cols <COL1> <COL2> ... \
  --categorical-cols <COL1> <COL2> ... \
  --n-trials 100

# 予測生成
python main.py predict --train-data <PATH> --test-data <PATH> \
  --target-col <COL> --id-col <COL> \
  --drop-cols <COL1> <COL2> ... \
  --categorical-cols <COL1> <COL2> ...

# 全パイプライン（train → predict）
python main.py all --train-data <PATH> --test-data <PATH> \
  --target-col <COL> --id-col <COL> \
  --drop-cols <COL1> <COL2> ... \
  --categorical-cols <COL1> <COL2> ...
```

---

## 🧪 Implementation Details

### データ前処理

- **削除列**: PassengerId（学習時）, Name, Ticket, Cabin
- **使用特徴量**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked（7列）
- **カテゴリ変数**: Sex, Embarked（category型に変換、LightGBMが自動処理）
- **欠損値処理**: Age、Embarked、Fareの欠損値はLightGBMが自動処理（補完不要）
- **エンコーディング**: LightGBMのネイティブなカテゴリ処理機能を使用（One-hot不要）

### モデル設定

- **アルゴリズム**: LightGBM (Gradient Boosting Decision Tree)
- **デフォルトパラメータ**:
  - `objective`: binary（2値分類）
  - `metric`: binary_logloss
  - `boosting_type`: gbdt
  - `learning_rate`: 0.05
  - `num_leaves`: 31
  - `feature_fraction`: 0.9
  - `random_state`: 42
- **Early Stopping**: 50ラウンド
- **交差検証**: Stratified 5-Fold CV

### ハイパーパラメータ最適化（Optuna）

- **フレームワーク**: Optuna v3.5.0
- **探索空間**:
  - `num_leaves`: [15, 63]
  - `learning_rate`: [0.01, 0.1]（log scale）
  - `feature_fraction`: [0.6, 1.0]
  - `bagging_fraction`: [0.6, 1.0]
  - `bagging_freq`: [1, 7]
  - `min_child_samples`: [5, 50]
  - `lambda_l1`: [0.0, 10.0]（L1正則化）
  - `lambda_l2`: [0.0, 10.0]（L2正則化）
- **試行回数**: 100（デフォルト）
- **サンプラー**: TPE (Tree-structured Parzen Estimator)
- **プルーナー**: MedianPruner（早期打ち切り）
- **目的関数**: 5-Fold CVの平均精度を最大化

### 特徴量重要度（訓練結果の例）

| Feature | Importance (Gain) |
|---------|-------------------|
| Sex     | 2718              |
| Fare    | 1240              |
| Age     | 1229              |
| Pclass  | 905               |
| SibSp   | 148               |
| Embarked| 127               |
| Parch   | 90                |

---

## 📊 Output Files

### models/lightgbm_model.txt
`train_model()` から生成される訓練済みモデル（テキスト形式）

### models/lightgbm_model.pkl
`optimize_hyperparameters()` から生成される最適化済みモデル（pickle形式）

### models/lightgbm_params.json
ベストパラメータとCV結果:
```json
{
  "best_params": {...},
  "cv_results": {
    "mean_accuracy": 0.8406,
    "std_accuracy": 0.0201,
    "fold_scores": [0.8318, 0.8360, ...]
  },
  "best_iteration": 156,
  "optuna_study": {...},
  "timestamp": "2026-01-12T20:20:00"
}
```

### processed_data/metadata.json
前処理のメタデータ:
```json
{
  "categorical_cols": ["Sex", "Embarked"],
  "feature_names": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
  "target_col": "Survived",
  "id_col": "PassengerId",
  "drop_cols": ["Name", "Ticket", "Cabin"]
}
```

### output/submission.csv
Kaggle提出用ファイル:
```csv
PassengerId,Survived
892,0
893,0
894,0
...
```

### 可視化ファイル（generate_plots=True時）

- **learning_curves.png**: 訓練/検証データの損失推移（5-Fold平均）
- **feature_importance.png**: 特徴量の重要度ランキング
- **partial_dependence_plots.png**: 各特徴量の部分依存（3x3グリッド）

---

## 🔄 Workflow Comparison

### ワークフロー1: 標準訓練（train）

**CLI:**
```bash
python main.py train
python main.py predict
```

**Jupyter:**
```python
from train import train_model
result = train_model(...)
```

- **用途**: 素早いモデル検証、ベースライン作成
- **出力**: `models/lightgbm_model.txt`
- **所要時間**: 約5-10秒

### ワークフロー2: 最適化訓練（optimize）

**CLI:**
```bash
python main.py optimize --n-trials 100
python main.py predict
```

**Jupyter:**
```python
from optimize import optimize_hyperparameters
result = optimize_hyperparameters(n_trials=100)
```

- **用途**: Kaggle提出用の最高性能モデル
- **出力**: `models/lightgbm_model.pkl`, `lightgbm_params.json`, `optuna_study.pkl`
- **所要時間**: 約10-20分

---

## 🧑‍💻 Development

### テストの実行

```bash
# 独立関数のテスト
python test_jupyter_functions.py
```

### 新しいデータセットへの適用

Jupyter Notebook用の関数は完全に汎用的で、任意のデータセットに適用可能:

```python
# カスタムデータセットの例
result = preprocess_data(
    train_path='data/custom_train.csv',
    test_path='data/custom_test.csv',
    target_col='Churn',                    # 自由に変更可能
    id_col='CustomerId',
    drop_cols=['Name', 'Email'],
    categorical_cols=['Gender', 'Country']
)

train_result = train_model(
    train_data_path='processed_data/processed_train.csv',
    target_col='Churn',
    id_col='CustomerId',
    categorical_cols=['Gender', 'Country']
)
```

---

## 📦 Requirements

### Python環境

- Python 3.11+
- pandas 2.1.4
- numpy 1.26.2
- scikit-learn 1.3.2
- lightgbm 4.1.0
- optuna 3.5.0
- joblib 1.3.2
- matplotlib 3.x
- seaborn 0.x

### Docker環境

- Docker 20.10+
- Docker Compose 1.29+

すべての依存関係は `requirements.txt` に記載されています。

---

## 🎓 Key Innovations

### 1. モジュール独立性
各モジュールの新関数（`preprocess_data()`, `train_model()`, `optimize_hyperparameters()`, `predict()`）は完全に独立しており、`src.*` からのインポートは不要です。

### 2. CSV経由のデータフロー
モジュール間は中間CSVファイルで接続され、各ステップを独立して実行・検証できます。

### 3. 柔軟な実行方法
同じコードベースでCLI（自動化向き）とJupyter（実験向き）の両方に対応。

### 4. メタデータ管理
`metadata.json` でカテゴリカル列等の情報を保存し、再現性を確保。

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [LightGBM](https://github.com/microsoft/LightGBM)
- [Optuna](https://optuna.org/)

---

## 📮 Contact

質問やフィードバックがあれば、GitHubのIssuesまでお願いします。
