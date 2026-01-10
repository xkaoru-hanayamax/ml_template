import argparse
import sys

from src.train import run_train
from src.predict import run_predict
from src.optimize import run_optimize


def main():
    """
    Titanic Survival Prediction CLI

    Usage:
        python main.py train       # 学習（K-Fold CV + モデル保存）
        python main.py predict     # 予測（submission.csv生成）
        python main.py all         # 全実行（train → predict）
    """
    parser = argparse.ArgumentParser(
        description='Titanic Survival Prediction with LightGBM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train       # Train model with K-Fold CV (quick validation)
  python main.py optimize    # Optimize hyperparameters with Optuna + train final model
  python main.py predict     # Generate predictions
  python main.py all         # Run train then predict
        """
    )
    parser.add_argument(
        'command',
        choices=['train', 'predict', 'optimize', 'all'],
        help='Command to execute'
    )

    args = parser.parse_args()

    try:
        if args.command == 'train':
            run_train()
        elif args.command == 'optimize':
            run_optimize()
        elif args.command == 'predict':
            run_predict()
        elif args.command == 'all':
            print("\nExecuting full pipeline: train → predict\n")
            run_train()
            print("\n")
            run_predict()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
