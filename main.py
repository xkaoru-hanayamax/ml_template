import argparse
import sys

from src.train import run_train
from src.predict import run_predict
from src.optimize import run_optimize
from src.config import DatasetConfig


def main():
    """
    Generic Machine Learning Pipeline CLI

    Usage:
        python main.py train --train-data ... --test-data ... --target-col ... --id-col ... --drop-cols ... [--categorical-cols ...]
        python main.py predict --train-data ... --test-data ... --target-col ... --id-col ... --drop-cols ... [--categorical-cols ...]
        python main.py optimize --train-data ... --test-data ... --target-col ... --id-col ... --drop-cols ... [--categorical-cols ...] [--n-trials N]
        python main.py all --train-data ... --test-data ... --target-col ... --id-col ... --drop-cols ... [--categorical-cols ...]
    """
    parser = argparse.ArgumentParser(
        description='Generic Machine Learning Pipeline with LightGBM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Titanic dataset
  python main.py train --train-data data/train.csv --test-data data/test.csv --target-col Survived --id-col PassengerId --drop-cols PassengerId Name Ticket Cabin --categorical-cols Sex Embarked

  # Custom dataset
  python main.py train --train-data data/custom_train.csv --test-data data/custom_test.csv --target-col Churn --id-col CustomerId --drop-cols CustomerId Name Email --categorical-cols Gender Country
        """
    )
    parser.add_argument(
        'command',
        choices=['train', 'predict', 'optimize', 'all'],
        help='Command to execute'
    )

    # データセット設定引数
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--target-col', type=str, required=True,
                       help='Name of target column')
    parser.add_argument('--id-col', type=str, required=True,
                       help='Name of ID column')
    parser.add_argument('--drop-cols', type=str, nargs='+', required=True,
                       help='Columns to drop from features (space-separated)')
    parser.add_argument('--categorical-cols', type=str, nargs='+', default=[],
                       help='Categorical feature columns (space-separated)')

    # 学習設定引数
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of Optuna trials for optimization (default: 100)')

    args = parser.parse_args()

    # 設定オブジェクトを作成
    config = DatasetConfig(
        train_path=args.train_data,
        test_path=args.test_data,
        target_col=args.target_col,
        id_col=args.id_col,
        drop_cols=args.drop_cols,
        categorical_cols=args.categorical_cols
    )

    try:
        if args.command == 'train':
            run_train(config=config)
        elif args.command == 'optimize':
            run_optimize(config=config, n_trials=args.n_trials)
        elif args.command == 'predict':
            run_predict(config=config)
        elif args.command == 'all':
            print("\nExecuting full pipeline: train → predict\n")
            run_train(config=config)
            print("\n")
            run_predict(config=config)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
