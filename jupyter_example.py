"""
Jupyter Notebook使用例
このファイルの各セルをJupyter Notebookにコピー＆ペーストして使用できます
"""

# ============================================================
# Cell 1: セットアップ
# ============================================================
import sys
sys.path.insert(0, 'src')  # srcディレクトリをパスに追加

# ============================================================
# Cell 2: データの前処理
# ============================================================
from preprocessor import preprocess_data, load_metadata

# データの前処理を実行
result = preprocess_data(
    train_path='data/train.csv',
    test_path='data/test.csv',
    target_col='Survived',
    id_col='PassengerId',
    drop_cols=['Name', 'Ticket', 'Cabin'],
    categorical_cols=['Sex', 'Embarked'],
    output_dir='processed_data'
)

print(f"\n前処理完了！")
print(f"訓練データ: {result['train_output']}")
print(f"テストデータ: {result['test_output']}")
print(f"メタデータ: {result['metadata']}")

# メタデータの読み込み（後で使用する場合）
metadata = load_metadata(result['metadata'])
print(f"\nカテゴリカル列: {metadata['categorical_cols']}")

# ============================================================
# Cell 3: モデルの学習（通常の訓練）
# ============================================================
from train import train_model

# モデルを訓練
train_result = train_model(
    train_data_path='processed_data/processed_train.csv',
    target_col='Survived',
    id_col='PassengerId',
    categorical_cols=['Sex', 'Embarked'],
    generate_plots=True,  # グラフを生成
    n_folds=5,  # 5-Fold CV
    num_boost_round=1000,
    early_stopping_rounds=50
)

print(f"\n学習完了！")
print(f"モデル: {train_result['model_path']}")
print(f"平均精度: {train_result['mean_accuracy']:.4f}")
print(f"精度の標準偏差: {train_result['std_accuracy']:.4f}")

# ============================================================
# Cell 3 (代替): Optunaによるハイパーパラメータ最適化
# ============================================================
# from optimize import optimize_hyperparameters
#
# # ハイパーパラメータを最適化してモデルを訓練
# opt_result = optimize_hyperparameters(
#     train_data_path='processed_data/processed_train.csv',
#     target_col='Survived',
#     id_col='PassengerId',
#     categorical_cols=['Sex', 'Embarked'],
#     n_trials=50,  # 試行回数（デフォルト: 100）
#     n_folds=5
# )
#
# print(f"\n最適化完了！")
# print(f"モデル: {opt_result['model_path']}")
# print(f"ベストスコア: {opt_result['best_score']:.4f}")
# print(f"ベストパラメータ: {opt_result['best_params']}")

# ============================================================
# Cell 4: 予測の実行
# ============================================================
from predict import predict

# テストデータに対して予測を実行
submission = predict(
    test_data_path='processed_data/processed_test.csv',
    model_path='models/lightgbm_model.txt',  # または 'models/lightgbm_model.pkl'
    id_col='PassengerId',
    target_col='Survived',
    categorical_cols=['Sex', 'Embarked'],
    output_path='output/submission.csv'
)

print(f"\n予測完了！")
print(f"予測結果の最初の10件:")
print(submission.head(10))

print(f"\n生存予測の分布:")
print(submission['Survived'].value_counts())

# ============================================================
# 完了！
# ============================================================
print("\n" + "=" * 70)
print("すべての処理が完了しました！")
print("=" * 70)
print("\n生成されたファイル:")
print("  - processed_data/processed_train.csv")
print("  - processed_data/processed_test.csv")
print("  - processed_data/metadata.json")
print("  - models/lightgbm_model.txt (または .pkl)")
print("  - output/submission.csv")
print("  - output/learning_curves.png")
print("  - output/feature_importance.png")
print("  - output/partial_dependence_plots.png")
