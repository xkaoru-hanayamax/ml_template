"""
Jupyter Notebook用の独立関数のテストスクリプト
"""

# sys.pathに追加してsrcディレクトリからインポートできるようにする
import sys
sys.path.insert(0, 'src')

from preprocessor import preprocess_data, load_metadata
from train import train_model
from predict import predict

print("=" * 70)
print("Testing Jupyter Notebook Functions")
print("=" * 70)

# Test 1: Preprocessor
print("\n" + "=" * 70)
print("TEST 1: preprocess_data()")
print("=" * 70)

try:
    result = preprocess_data(
        train_path='data/train.csv',
        test_path='data/test.csv',
        target_col='Survived',
        id_col='PassengerId',
        drop_cols=['Name', 'Ticket', 'Cabin'],
        categorical_cols=['Sex', 'Embarked'],
        output_dir='processed_data_test'
    )
    print("\n[OK] preprocess_data() completed successfully!")
    print(f"  Train output: {result['train_output']}")
    print(f"  Test output: {result['test_output']}")
    print(f"  Metadata: {result['metadata']}")

    # Test metadata loading
    metadata = load_metadata(result['metadata'])
    print(f"\n[OK] load_metadata() completed successfully!")
    print(f"  Metadata keys: {list(metadata.keys())}")
    print(f"  Categorical cols: {metadata['categorical_cols']}")

except Exception as e:
    print(f"\n[FAIL] preprocess_data() FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Train Model
print("\n" + "=" * 70)
print("TEST 2: train_model()")
print("=" * 70)

try:
    train_result = train_model(
        train_data_path='processed_data_test/processed_train.csv',
        target_col='Survived',
        id_col='PassengerId',
        categorical_cols=['Sex', 'Embarked'],
        generate_plots=False,  # Skip plots for faster testing
        model_output_dir='models_test',
        n_folds=2,  # Reduce folds for faster testing
        num_boost_round=50,  # Reduce rounds for faster testing
        early_stopping_rounds=10
    )
    print("\n[OK] train_model() completed successfully!")
    print(f"  Model path: {train_result['model_path']}")
    print(f"  Mean accuracy: {train_result['mean_accuracy']:.4f}")
    print(f"  Std accuracy: {train_result['std_accuracy']:.4f}")
    print(f"  CV scores: {train_result['cv_scores']}")

except Exception as e:
    print(f"\n[FAIL] train_model() FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Predict
print("\n" + "=" * 70)
print("TEST 3: predict()")
print("=" * 70)

try:
    submission = predict(
        test_data_path='processed_data_test/processed_test.csv',
        model_path='models_test/lightgbm_model.txt',
        id_col='PassengerId',
        target_col='Survived',
        categorical_cols=['Sex', 'Embarked'],
        output_path='output_test/submission.csv'
    )
    print("\n[OK] predict() completed successfully!")
    print(f"  Submission shape: {submission.shape}")
    print(f"  Submission columns: {list(submission.columns)}")
    print(f"\nFirst 5 predictions:")
    print(submission.head())

except Exception as e:
    print(f"\n[FAIL] predict() FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
